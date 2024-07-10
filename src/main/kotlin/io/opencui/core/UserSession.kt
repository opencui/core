package io.opencui.core

import java.util.*
import kotlin.collections.LinkedHashMap
import kotlin.reflect.KClass
import kotlin.reflect.full.primaryConstructor
import io.opencui.serialization.*
import com.fasterxml.jackson.annotation.JsonIgnore
import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.node.*
import io.opencui.core.user.IUserIdentifier
import io.opencui.core.user.UserIdentifier
import io.opencui.core.da.DialogAct
import io.opencui.core.da.FrameDialogAct
import io.opencui.core.da.SlotDialogAct
import io.opencui.du.ListRecognizer
import io.opencui.kvstore.IKVStore
import io.opencui.sessionmanager.ChatbotLoader
import io.opencui.system1.CoreMessage
import java.io.ObjectInputStream
import java.io.Serializable
import java.time.Duration
import java.time.LocalDateTime
import java.util.concurrent.ConcurrentHashMap
import kotlin.reflect.KParameter
import org.slf4j.Logger
import org.slf4j.LoggerFactory

//
// Scheduler holds fillers in the single tree.
//
class Scheduler(val session: UserSession): ArrayList<IFiller>(), Serializable {
    // TODO(xiaobo): what is the difference between reschedule and recover, and where is the confirmation, vr, vc?
    enum class State {
        INIT,
        ASK,
        POST_ASK,
        RESCHEDULE,
        RESPOND,
        RECOVER,
    }
    enum class Side {
        INSIDE,
        OUTSIDE,
    }
    var state: State = State.INIT
    var side: Side = Side.INSIDE

    fun push(item: IFiller) {
        logger.debug("pushing ${item::class.java} with ${item.path?.last()}")
        add(item)
        item.onPush()
    }

    fun pop(): IFiller {
        val item = lastOrNull()
        if (!isEmpty()) {
            removeAt(size - 1)
        }
        item!!.onPop()
        logger.debug("popping ${item::class.java} with ${item.path?.last()}")
        return item
    }

    fun peek(): IFiller = last()

    fun parentGrandparentBothAnnotated(): Boolean {
        if (size < 3) return false
        return get(size - 1) is FrameFiller<*> && get(size - 2) is AnnotatedWrapperFiller && get(size - 3) is AnnotatedWrapperFiller
    }

    /**
     * This is used when first expand the system. This call is guaranteed to work for
     * first time call on correct intent definition.
     * This make sure that, there are stuff need to be done
     */
    fun grow(): Boolean {
        var top = this.peek()

        while (!top.move(session, session.activeEvents)) {
            // Find some open composite to put to top, so that we have more things to work with.
            val grown = if (top is ICompositeFiller) top.grow(session, session.activeEvents) else false
            if (!grown) return false
            top = this.peek()
        }
        return true
    }

    fun cleanup() {
        clear()
        state = State.INIT
    }

    fun toObjectNode(): ObjectNode {
        val objNode = ObjectNode(JsonNodeFactory.instance)
        objNode.replace("state", TextNode(state.name))
        val nodeArray = mutableListOf<JsonNode>()
        for (filler in this) {
            val node = ObjectNode(JsonNodeFactory.instance)
            node.replace("filler_type", TextNode(filler.javaClass.simpleName))
            node.replace("attribute", TextNode(filler.attribute))
            nodeArray.add(node)
        }
        objNode.replace("fillers", ArrayNode(JsonNodeFactory.instance, nodeArray))
        return objNode
    }

    companion object {
        val logger = LoggerFactory.getLogger(Scheduler::class.java)
    }
}

/**
 * The core of the interaction is a statechart where we have session level,
 * frame level (nested) and slot level decision need to make. And these statechart are
 * static in nature, but its behavior is dynamically decided by guard on the transition.
 *
 * It should be useful for UserSession to host the statechart at session level.
 *
 * At session level, we only need to do a couple of things:
 * a. when there is no active skill.
 *    1. where there is skill event.
 *    2. where there si no meaningful event.
 *    3. where there is slot events that can be fixed by constrained intent suggestions.
 * b. where there is active skill:
 *    1. where there is new skill event:
 *    2. where there is no new skill events:
 *       i. there is no event
 *       ii. there is slot event that is not compatible.
 *       iii. there is slot event that is compatible.
 *
 */
interface StateChart {
    // This host all the events currently not fully consumed.
    val events : MutableList<FrameEvent>
    var turnId: Int

    // This ensures the parallelism of the StartChart at the top level.
    val schedulers: MutableList<Scheduler>

    val activeEvents: List<FrameEvent>
        get() = events.filter { !it.usedUp }

    val mainSchedule : Scheduler
        get() = schedulers.first()

    val schedule: Scheduler
        get() = schedulers.last()

    fun tryToRefocus(frameEvents: List<FrameEvent>): Pair<List<IFiller>, FrameEvent>?

    fun getActionByEvent(frameEvents: List<FrameEvent>): Action?

    val finishedIntentFiller: MutableList<AnnotatedWrapperFiller>
    
    fun addEvent(frameEvent: FrameEvent)
    fun addEvents(frameEvents: List<FrameEvent>)

    /**
     * Take one or more steps of statechart transition.
     * The main constraint is we can not have turnComplete in the middle of returned action list.
     *
     * 1. Based on current states of the chart forest.
     * 2. Take the most relevant FrameEvent.
     * 3. Build the action need to be executed at this state, which updates the state.
     *
     * There are three types of atomic actions:
     * Chart building, state transition, and message emission, of course, there are composite ones.
     */
    fun kernelStep(): List<Action>
}


/**
 * UserSession is used to keep the history of the conversation with user. It will
 * only keep for certain amount of time.
 *
 * In the UserSession, we keep global value, and current intent, and filling schedule,
 * explanations left over, and background intent. Basically all the information needed
 * for continue the session.
 */
data class UserSession(
    override var userId: String?,
    override var channelType: String? = null,
    override var channelLabel: String? = null,
    @Transient @JsonIgnore var chatbot: IChatbot? = null,

): LinkedHashMap<String, Any>(), Serializable, StateChart, IUserIdentifier {

    override var isVerfied: Boolean = false
    override var name: PersonName? = null
    override var phone: PhoneNumber? = null
    override var email: Email? = null

    constructor(u: IUserIdentifier, c: IChatbot?): this(u.userId, u.channelType, u.channelLabel, c)

    // Default botInfo, need to be changed.
    val botInfo : BotInfo by lazy { botInfo(chatbot!!.orgName, chatbot!!.agentName, chatbot!!.agentLang, chatbot!!.agentBranch) }

    override val events = mutableListOf<FrameEvent>()

    val userIdentifier: IUserIdentifier
        get() = this

    fun getLocale(): Locale {
        // For now, we use language to determine locale, but we can accept the client suggested value later.
        return Locale(botInfo.lang)
    }


    override fun addEvent(frameEvent: FrameEvent) {
        frameEvent.updateTurnId(turnId)
        if (frameEvent.source == EventSource.API) {
            events.removeIf{ it.fullType == frameEvent.fullType }
        }
        events.add(frameEvent)
    }

    val history = mutableListOf<CoreMessage>()

    // for now, use 30 minutes senssion
    val sessionDuration = 30*60
    val msgIds = ConcurrentHashMap<String, LocalDateTime>()

    @Transient
    var turnRecognizer: ListRecognizer? = null
    var sessionRecognizer: ListRecognizer? = null

    @Transient
    override var messageId: String? = null

    override var sessionId: String? = null

    // This function try to check whether the message is the first
    // The idea is we only process the first message in the sequence of
    // retries.
    fun isFirstMessage(pmsgId: String): Boolean {
        // First remove old msgIds.
        val now = LocalDateTime.now()
        for (msgId in msgIds.keys) {
            val duration = Duration.between(msgIds[msgId]!!, now)
            if (duration.seconds > sessionDuration) {
                msgIds.remove(msgId)
            }
        }

        // For now, we only process once.
        return if (!msgIds.containsKey(pmsgId)) {
            msgIds.put(pmsgId, LocalDateTime.now())
            true
        } else {
            false
        }
    }

    override fun addEvents(frameEvents: List<FrameEvent>) {
        frameEvents.forEach { addEvent(it) }
    }

    // the timezone is session dependent. For example, when user ask about New York hotel, then ask the same
    // thing about san fransisco.
    var timezone : String? = null

    fun addUserMessage(msg: String) {
        history.add(CoreMessage(true, msg))
    }

    fun addBotMessage(msg: String) {
        history.add(CoreMessage(false, msg))
    }

    var targetChannel: List<String> = listOf(SideEffect.RESTFUL)

    @JsonIgnore
    override val schedulers: MutableList<Scheduler> = mutableListOf(Scheduler(this))

    @JsonIgnore
    override var turnId: Int = 0

    @JsonIgnore
    var lastTurnRes: List<ActionResult> = listOf()

    // For now, we assume that only default locale for each language is used.
    val rgLang : RGBase
        get() = chatbot!!.duMeta.getRGLang("")

    /**
     * Chart building are in kernel mode and should not be exposed to execution.
     */
    fun userStep(): List<Action> {
        var res = kernelStep()
        while (res.size == 1 && (res[0] is KernelMode)) {
            res[0].wrappedRun(this)
            res = kernelStep()
        }
        // make sure there is no chart building action leak to user space.
        assert(res.none { it is KernelMode })
        return res
    }


    override fun kernelStep(): List<Action> {
        // system-driven process
        if (schedule.state == Scheduler.State.ASK) {
            return listOf(SlotAskAction())
        }

        if (schedule.state == Scheduler.State.POST_ASK) {
            val currentFiller = schedule.lastOrNull()
            if (currentFiller != null) {
                val events = activeEvents.sortedBy { if (it.refocused) 0 else 1 }
                val strictlyMatch = events.firstOrNull { currentFiller.isCompatible(it) }
                if (strictlyMatch != null) {
                    return listOf(SlotPostAskAction(currentFiller, strictlyMatch))
                }
                for (event in events.filter { it.turnId == turnId && !it.inferredFrom && !it.isUsed }) {
                    val inferredMatch = (currentFiller as? Infer)?.infer(event)
                    if (inferredMatch != null) {
                        event.inferredFrom = true
                        event.triggered = true
                        return listOf(SlotPostAskAction(currentFiller, inferredMatch))
                    }
                }
            }
        }

        if (schedule.state == Scheduler.State.RESPOND) {
            val currentFiller = schedule.lastOrNull()
            check(currentFiller is AnnotatedWrapperFiller)
            return listOf(RespondAction())
        }

        // user-driven process

        // state update
        val eventTriggeredTransition = getActionByEvent(activeEvents.filter { it.turnId == turnId && !it.triggered && !it.isUsed })
        if (eventTriggeredTransition != null) {
            return listOf(eventTriggeredTransition)
        }

        // special matcher for HasMore with PagedSelectable in it
        val refocusPair = tryToRefocus(activeEvents.filter { it.turnId == turnId && !it.refocused && !it.isUsed})
        // prevent from refocusing from kernel mode to user mode
        if (refocusPair != null && (!inKernelMode(schedule) || inKernelMode(refocusPair.first))) {
            val refocusFiller = refocusPair.first.last() as AnnotatedWrapperFiller
            return if (
                !refocusFiller.targetFiller.done(emptyList()) &&
                (refocusFiller.targetFiller is EntityFiller<*> || refocusFiller.targetFiller is OpaqueFiller<*>)
                ) {
                listOf(SimpleFillAction(refocusFiller.targetFiller as AEntityFiller, refocusPair.second))
            } else {
                if ((refocusPair.first.last() as? AnnotatedWrapperFiller)?.targetFiller !is MultiValueFiller<*>) {
                    // view refocusing to multi-value slot as adding value, others as starting over
                    refocusPair.first.last().clear()
                }
                refocusPair.second.refocused = true
                listOf(RefocusAction(refocusPair.first))
            }
        }

        val frameEvent = activeEvents.filter { it.turnId == turnId && !it.triggered && !it.isUsed && !it.refocused && !it.inferredFrom }.firstOrNull()
        // new scheduler for new intent
        if (frameEvent != null) {
            val type = frameEvent.type
            if (type == "" && schedule.isEmpty()) {
                val fullyQualifiedName: String = SystemAnnotationType.IntentSuggestion.typeName
                if (!fullyQualifiedName.isEmpty()) {
                    if (fullyQualifiedName.lastIndexOf(".") >= 0 ) {
                        return listOf(StartFill(frameEvent, intentBuilder(fullyQualifiedName), "systemAnnotation"))
                    }
                }
            }

            if (isOpenIntent(frameEvent)) {
                // if it is supposed to trigger new intent, but it does not trigger it based on our rules,
                // it is not allowed to trigger new intent in the following turns
                frameEvent.triggered = true
                frameEvent.typeUsed = true
            } else {
                val buildIntent = EventFrameBuilder(frameEvent)
                if (buildIntent.invoke(this) != null) {
                    return listOf(StartFill(frameEvent, buildIntent, "construct"))
                }
            }
        }

        if (schedule.state == Scheduler.State.RESCHEDULE) {
            return listOf(RescheduleAction())
        }

        // recover process
        if (schedule.state == Scheduler.State.RECOVER) {
            return listOf(RecoverAction())
        }
        return listOf()
    }

    /**
     * This is the new way of storing session global information, where
     * we identify things by fully qualified name.
     */
    @JsonIgnore
    val globals = LinkedHashMap<String, ISingleton>()

    @JsonIgnore
    override val finishedIntentFiller = mutableListOf<AnnotatedWrapperFiller>()

    // for support only.
    var botOwn: Boolean = true

    fun searchContext(candidateClass: List<String>): List<Any> {
        val result = mutableListOf<Any>()
        val candidatesFillers =
                mainSchedule.reversed().filterIsInstance<AnnotatedWrapperFiller>().filter { it.targetFiller is FrameFiller<*> } +
                mainSchedule.reversed().filterIsInstance<MultiValueFiller<*>>().filter { it.svType == MultiValueFiller.SvType.INTERFACE }.flatMap { it.fillers.mapNotNull { (it.targetFiller as? InterfaceFiller<*>)?.vfiller } } +
                finishedIntentFiller.toList().filter { (it.targetFiller as TypedFiller<*>).target.get() !is AbstractValueClarification<*> }
        val contextValueCandidates: List<Any> = candidatesFillers.flatMap {
            val res = mutableListOf<Any>()
            res.add((it.targetFiller as FrameFiller<*>).target.get()!!)
            res.addAll(it.targetFiller.fillers.values.filter { it.targetFiller is AEntityFiller || it.targetFiller is FrameFiller<*> }
                    .mapNotNull { (it.targetFiller as? TypedFiller<*>)?.target?.get() })
            res.addAll(it.targetFiller.fillers.values.filter { it.targetFiller is MultiValueFiller<*> }.flatMap {
                (it.targetFiller as MultiValueFiller<*>).fillers.mapNotNull { (it.targetFiller as? TypedFiller<*>)?.target?.get() } })
            res
        }.toSet().toList()
        val cachedKClass = mutableMapOf<String, KClass<*>>()
        for (c in candidateClass) {
            if (!cachedKClass.containsKey(c)) {
                try {
                    val kClass = findKClass(c)!!
                    cachedKClass[c] = kClass
                } catch (e: Exception) {
                    e.printStackTrace()
                }
            }
        }
        for (target in contextValueCandidates) {
            for (c in candidateClass) {
                if (c == target::class.qualifiedName || (cachedKClass.containsKey(c) && cachedKClass[c]!!.isInstance(target))) {
                    result.add(target)
                }
            }
        }
        return result
    }

    // decide whether the active intent is IKernelIntent
    fun inKernelMode(s: List<IFiller>): Boolean {
        return s.any { it is FrameFiller<*> && it.frame() is IKernelIntent }
    }

    inline fun <reified T : IExtension> getExtension() : T? {
        val kClass = T::class.java
        if (kClass.isAssignableFrom(IKVStore::class.java)) {
            return Dispatcher.sessionManager.botStore as T
        }

        // We always try to clone for session, but default implementation does nothing.
        val res = chatbot!!.extensions.get<T>()?.cloneForSession(this) ?: return null
        if (res is IProvider) {
            res.session = this
        }
        return res as T
    }

    fun generateFrameEvent(filler: IFiller, value: Any): List<FrameEvent> {
        val fullyQualifiedType: String = filler.qualifiedEventType() ?: if (value is ObjectNode) value.get("@class").asText() else value::class.qualifiedName!!
        val typeString = fullyQualifiedType.substringAfterLast(".")
        val packageName = fullyQualifiedType.substringBeforeLast(".")
        val result = if (value is ObjectNode) {
            value.remove("@class")
            when (value) {
                is ValueNode -> {
                    listOf(
                        FrameEvent.fromJson(typeString, Json.makeObject(mapOf(filler.attribute to value))).apply {
                            this.packageName = packageName
                        })
                }

                is ObjectNode -> {
                    listOf(FrameEvent.fromJson(typeString, value).apply {
                        this.packageName = packageName
                    })
                }

                is ArrayNode -> {
                    value.mapNotNull {
                        when (it) {
                            is ValueNode -> {
                                FrameEvent.fromJson(typeString, Json.makeObject(mapOf(filler.attribute to it))).apply {
                                    this.packageName = packageName
                                }
                            }
                            is ObjectNode -> {
                                FrameEvent.fromJson(typeString, it).apply {
                                    this.packageName = packageName
                                }
                            }
                            else -> {
                                null
                            }
                        }
                    }
                }
                else -> {
                    listOf()
                }
            }
        } else {
            val jsonElement = Json.encodeToJsonElement(value)
            when {
                filler is OpaqueFiller<*> -> {
                    val declaredType = filler.declaredType
                    val nestedTypeString = declaredType.substringAfterLast(".")
                    val nestedPackageName = declaredType.substringBeforeLast(".")
                    val nestedFrames = FrameEvent.fromJson(nestedTypeString, jsonElement).apply {
                        this.packageName = nestedPackageName
                    }
                    nestedFrames.attribute = filler.attribute
                    listOf(FrameEvent(typeString, listOf(), listOf(nestedFrames)).apply { this.packageName = packageName })
                }
                jsonElement is ValueNode || value is IEntity -> {
                    listOf(
                        FrameEvent.fromJson(typeString, Json.makeObject(mapOf(filler.attribute to jsonElement))).apply {
                            this.packageName = packageName
                        })
                }
                jsonElement is ObjectNode -> {
                    listOf(FrameEvent.fromJson(typeString, jsonElement).apply {
                        this.packageName = packageName
                    })
                }
                else -> {
                    listOf()
                }
            }
        }

        logger.debug("generated event: $result")
        return result
    }

    fun findWrapperFillerForTargetSlot(frame: IFrame, nested_slot: String?): AnnotatedWrapperFiller? {
        var filler = findWrapperFillerWithFrame(frame)
        if (nested_slot.isNullOrEmpty() || nested_slot == "this") return filler

        // For nested slot, we move down the chain.
        val slots = nested_slot.split(".")
        for (slot in slots) {
            filler = (filler?.targetFiller as? FrameFiller<*>)?.fillers?.get(slot) ?: return null
        }
        return filler
    }

    fun findWrapperFillerWithFrame(frame: IFrame): AnnotatedWrapperFiller? {
        for (s in schedulers) {

            // search in all builder defined slots first
            val first = s.firstOrNull() ?: continue
            val path = findFillerPath(first) { f -> f is AnnotatedWrapperFiller && f.targetFiller is FrameFiller<*> && f.targetFiller.frame() === frame }
            if (path.isNotEmpty()) return path.last() as AnnotatedWrapperFiller

            // search in all active fillers including fillers for VR, VC and Confirmation
            for (f in s) {
                if (f is AnnotatedWrapperFiller && f.targetFiller is FrameFiller<*> && f.targetFiller.frame() === frame) {
                    return f
                }
            }
        }
        return null
    }

    fun findActiveFillerPathForTargetSlot(frame: IFrame, slot: String?): LinkedList<IFiller> {
        val path = findActiveFillerPathByFrame(frame)
        if (path.isNotEmpty() && !slot.isNullOrEmpty() && slot != "this") {
            val frameFiller = (path.last() as? AnnotatedWrapperFiller)?.targetFiller as? FrameFiller<*>
            val slotFiller = frameFiller?.get(slot) as? AnnotatedWrapperFiller
            if (frameFiller != null && slotFiller != null) {
                path += frameFiller
                path += slotFiller
            }
        }
        return path
    }

    // only finds active fillers, that is fillers direct in the stack including VR, VC and Confirmation fillers
    fun findActiveFillerPathByFrame(frame: IFrame): LinkedList<IFiller> {
        val path: LinkedList<IFiller> = LinkedList()
        for (s in schedulers) {
            for (f in s) {
                if (f is AnnotatedWrapperFiller && f.targetFiller is FrameFiller<*> && f.targetFiller.frame() === frame) {
                    val index = s.indexOf(f)
                    path.addAll(s.subList(0, index+1))
                    return path
                }
            }
        }
        return path
    }

    fun isRightMostChild(p: AnnotatedWrapperFiller, c: AnnotatedWrapperFiller): Boolean {
        val targetFiller = p.targetFiller
        if (targetFiller is AEntityFiller) {
            return false
        } else if (targetFiller is FrameFiller<*>) {
            return c === targetFiller.fillers.values.lastOrNull()
        } else if (targetFiller is InterfaceFiller<*>) {
            return c === targetFiller.vfiller
        } else if (targetFiller is MultiValueFiller<*>) {
            return c === targetFiller.fillers.lastOrNull()
        }
        return false
    }

    fun hasNoChild(p: AnnotatedWrapperFiller): Boolean {
        val targetFiller = p.targetFiller
        if (targetFiller is AEntityFiller) {
            return true
        } else if (targetFiller is FrameFiller<*>) {
            return targetFiller.fillers.isEmpty()
        } else if (targetFiller is InterfaceFiller<*>) {
            return targetFiller.vfiller == null
        } else if (targetFiller is MultiValueFiller<*>) {
            return targetFiller.fillers.isEmpty()
        }
        return true
    }

    fun pushAllChildren(p: AnnotatedWrapperFiller, stack: Stack<AnnotatedWrapperFiller>) {
        val targetFiller = p.targetFiller
        if (targetFiller is FrameFiller<*>) {
            targetFiller.fillers.values.reversed().forEach { stack.push(it) }
        } else if (targetFiller is InterfaceFiller<*>) {
            stack.push(targetFiller.vfiller!!)
        } else if (targetFiller is MultiValueFiller<*>) {
            targetFiller.fillers.reversed().forEach { stack.push(it) }
        }
    }

    fun postOrderManipulation(scheduler: Scheduler, start: AnnotatedWrapperFiller, end: AnnotatedWrapperFiller, task: (AnnotatedWrapperFiller) -> Unit) {
        val root = scheduler.firstOrNull() as? AnnotatedWrapperFiller ?: return
        val stack = Stack<AnnotatedWrapperFiller>()
        stack.push(root)
        var last: AnnotatedWrapperFiller? = null
        var started = false
        while (stack.isNotEmpty()) {
            val top = stack.peek()
            if ((last != null && isRightMostChild(top, last)) || hasNoChild(top)) {
                stack.pop()
                started =  started || top === start
                if (started) {
                    task(top)
                }
                last = top
                if (top === end) return
            } else {
                pushAllChildren(top, stack)
            }
        }
    }

    // only finds slot fillers; VR, VC, Confirmation fillers are filtered out
    fun findFillers(current: AnnotatedWrapperFiller?, res: MutableList<AnnotatedWrapperFiller>, filter: (AnnotatedWrapperFiller) -> Boolean, additionalBaseCase: (AnnotatedWrapperFiller) -> Boolean = { _ -> false}) {
        if (current == null || additionalBaseCase(current)) return
        if (filter(current)) {
            res += current
        }
        if (current.targetFiller is InterfaceFiller<*>) {
            findFillers(current.targetFiller.vfiller, res, filter, additionalBaseCase)
        } else if (current.targetFiller is FrameFiller<*>) {
            for (f in current.targetFiller.fillers.values) {
                findFillers(f, res, filter, additionalBaseCase)
            }
        } else if (current.targetFiller is MultiValueFiller<*>) {
            for (f in current.targetFiller.fillers) {
                findFillers(f, res, filter, additionalBaseCase)
            }
        }
    }

    // find filler path; VR, VC, Confirmation fillers are filtered out
    fun findFillerPath(current: IFiller?, filter: (IFiller) -> Boolean): LinkedList<IFiller> {
        var path: LinkedList<IFiller> = LinkedList()
        if (current == null) return path
        if (filter(current)) {
            path.offerFirst(current)
            return path
        }
        if (current is AnnotatedWrapperFiller) {
            path = findFillerPath(current.targetFiller, filter)
        } else if (current is InterfaceFiller<*>) {
            path = findFillerPath(current.vfiller, filter)
        } else if (current is FrameFiller<*>) {
            for (f in current.fillers.values) {
                path = findFillerPath(f, filter)
                if (path.isNotEmpty()) break
            }
        } else if (current is MultiValueFiller<*>) {
            for (f in current.fillers) {
                path = findFillerPath(f, filter)
                if (path.isNotEmpty()) break
            }
        }
        if (path.isNotEmpty()) path.offerFirst(current)
        return path
    }

    fun construct(packageName: String?, className: String, vararg args: Any?): IFrame? {
        val revisedPackageName = packageName ?: chatbot?.javaClass?.packageName
        try {
            val kClass = Class.forName("${revisedPackageName}.${className}", true, chatbot!!.getLoader()).kotlin
            val ctor = kClass.primaryConstructor ?: return null
            // Checking whether this is singleton.
            return ctor.call(*args) as? IFrame
        } catch (e: Exception) {
            e.printStackTrace()
            return null
        } catch (e: Error) {
            return null
        }
    }


    fun findKClass(className: String): KClass<*>? {
        return try {
            when (className) {
                "kotlin.Int" -> Int::class
                "kotlin.Float" -> Float::class
                "kotlin.String" -> String::class
                "kotlin.Boolean" -> Boolean::class
                else -> {
                    val kClass = Class.forName(className, true, chatbot!!.getLoader()).kotlin
                    kClass
                }
            }
        } catch (e: Exception) {
            null
        }
    }

    // refocus here means refocus on both filled slots and unfilled slots; depends on conditions
    // search from top of stack and involve children within one level from active filler
    override fun tryToRefocus(frameEvents: List<FrameEvent>): Pair<List<IFiller>, FrameEvent>? {
        if (schedule.isEmpty()) return null
        val stack: LinkedList<IFiller> = LinkedList()
        for (f in schedule) {
            stack.offerLast(f)
        }

        var last: IFiller? = null
        while (stack.isNotEmpty()) {
            val top = stack.peekLast() as? AnnotatedWrapperFiller
            if (top == null || top.targetFiller !is FrameFiller<*>) {
                stack.pollLast()
                continue
            }
            searchForRefocusChild(top.targetFiller, frameEvents, 0)?.let {
                stack.offerLast(top.targetFiller)
                stack.offerLast(it.first)
                return Pair(stack, it.second)
            }

            for (c in top.targetFiller.fillers.values.filter { it.targetFiller is FrameFiller<*> && it != last }) {
                searchForRefocusChild(c.targetFiller as FrameFiller<*>, frameEvents, 1)?.let {
                    stack.offerLast(top.targetFiller)
                    stack.offerLast(c)
                    stack.offerLast(c.targetFiller)
                    stack.offerLast(it.first)
                    return Pair(stack, it.second)
                }
            }
            last = stack.pollLast()
        }
        return null
    }

    private fun searchForRefocusChild(
        parent: FrameFiller<*>,
        frameEvents: List<FrameEvent>,
        level: Int
    ): Pair<AnnotatedWrapperFiller, FrameEvent>? {
        // conditions on which we allow refocus;
        // 1. filled slots of all candidate frames
        // 2. unfinished mv filler with HasMore FrameEvent (especially for the case in which we focus on PagedSelectable and user wants to say no to mv slot)
        // 3. unfilled slots of active frames (level == 0)
        //  (1) prevent the expression "to shanghai" from triggering new Intent
        //  (2) refocus to skill: IIntent (Intent Suggestion) excluding partially filled Interface type
        //  (3) refocus to entry filler of MultiValueFiller after we infer a hasMore.Yes. Excluding refocus to MultiValueFiller if there is entry filler not done
        val matcher: (AnnotatedWrapperFiller) -> Boolean = {
            val targetFiller = it.targetFiller
            it.canEnter(frameEvents)
            && (
                (targetFiller is AEntityFiller && targetFiller.done(emptyList())
                        && frameEvents.firstOrNull { e -> it.isCompatible(e) } != null)
                // special matcher for HasMore with PagedSelectable; allow to refocus to undone HasMore if there is one
                || (targetFiller is InterfaceFiller<*> &&
                        (it.parent as? FrameFiller<*>)?.frame() is HasMore
                        && (it.parent?.parent?.parent as? MultiValueFiller<*>)?.done(emptyList()) == false
                        && frameEvents.firstOrNull { e -> it.isCompatible(e) } != null)
                || (level == 0
                    && (targetFiller is AEntityFiller
                        || (targetFiller is InterfaceFiller<*> && targetFiller.realtype == null)
                        || (targetFiller is MultiValueFiller<*> && targetFiller.findCurrentFiller() == null))
                    && !targetFiller.done(frameEvents) && frameEvents.firstOrNull { e -> it.isCompatible(e) } != null)
            )
        }
        // open slots take priority
        val groups = parent.fillers.values.groupBy { it.done(frameEvents) }
        val filler = groups[false]?.firstOrNull(matcher)
                ?: groups[true]?.firstOrNull(matcher)
                ?: return null
        val e = frameEvents.first { e -> filler.isCompatible(e) }
        return Pair(filler, e)
    }

    override fun getActionByEvent(frameEvents: List<FrameEvent>): Action? {
        for (f in schedule.reversed()) {
            if (f is AnnotatedWrapperFiller && f.targetFiller is FrameFiller<*>) {
                val contextFrame = f.targetFiller.frame()
                for (event in frameEvents) {
                    val stateUpdateIntentBuilder = contextFrame.searchStateUpdateByEvent(event.fullType)
                    if (stateUpdateIntentBuilder != null) return StartFill(event, stateUpdateIntentBuilder, "stateupdate")
                }
            }
        }
        return null
    }

    fun findSystemAnnotation(systemAnnotationType: SystemAnnotationType, vararg args: Any?): IIntent? {
        val fullyQualifiedName: String = systemAnnotationType.typeName
        if (fullyQualifiedName.isEmpty()) return null
        val index = fullyQualifiedName.lastIndexOf(".")
        if (index < 0) return null
        val packageName = fullyQualifiedName.substring(0, index)
        val className = fullyQualifiedName.substring(index + 1)
        return construct(packageName, className, this, *args) as? IIntent
    }

    private fun genGroupKey(dialogAct: DialogAct): String {
        return when (dialogAct) {
            is SlotDialogAct -> """${if (dialogAct.context.isNotEmpty()) dialogAct.context.first()::class.qualifiedName else "null"}_${dialogAct.slotName}_${dialogAct.slotType}"""
            is FrameDialogAct -> dialogAct.frameType
            else -> dialogAct::class.qualifiedName!!
        }
    }

    private fun areParametersCompatible(formalParams: List<KParameter>, actualParams: List<DialogAct>): Boolean {
        check(formalParams.size == actualParams.size)
        for (i in formalParams.indices) {
            if ((formalParams[i].type.classifier as? KClass<*>)?.isInstance(actualParams[i]) != true) return false
        }
        return true
    }

    private fun findDialogActCustomization(dialogAct: DialogAct): Templates? {
        if (dialogAct is SlotDialogAct) {
            val annotations = dialogAct.context.firstOrNull()?.findAll<DialogActCustomizationAnnotation>(dialogAct.slotName) ?: listOf()
            return annotations.firstOrNull { it.dialogActName == dialogAct::class.qualifiedName }?.templateGen?.invoke(dialogAct)
        } else if (dialogAct is FrameDialogAct) {
            val packageName = dialogAct.frameType.substringBeforeLast(".")
            val className = dialogAct.frameType.substringAfterLast(".")
            val annotations = construct(packageName, className, this)?.findAll<DialogActCustomizationAnnotation>("this") ?: listOf()
            return annotations.firstOrNull { it.dialogActName == dialogAct::class.qualifiedName }?.templateGen?.invoke(dialogAct)
        } else {
            throw Exception("ComponentDialogAct not supported")
        }
    }

    private fun rewriteDialogActInGroup(group: List<DialogAct>): List<DialogAct> {
        val res = mutableListOf<DialogAct>()
        val constructors = chatbot!!.rewriteRules.map { it.primaryConstructor!! }.sortedByDescending { it.parameters.size }
        var index = 0
        while (index < group.size) {
            var combined: Boolean = false
            for (constructor in constructors) {
                if (index + constructor.parameters.size > group.size) continue
                if (areParametersCompatible(constructor.parameters, group.subList(index, index+constructor.parameters.size))) {
                    val r = (constructor.call(*group.toTypedArray())).invoke()
                    findDialogActCustomization(r)?.let {
                        r.templates = it
                    }
                    res += r
                    index += constructor.parameters.size
                    combined = true
                    break
                }
            }
            if (!combined) {
                res += group[index]
                index++
            }
        }
        return res
    }

    fun rewriteDialogAct(dialogActList: List<DialogAct>): List<DialogAct> {
        val groups: MutableList<Pair<String, MutableList<DialogAct>>> = mutableListOf()
        for (dialogAct in dialogActList) {
            val  key = genGroupKey(dialogAct)
            if (groups.isEmpty() || groups.last().first != key) {
                groups += Pair(key, mutableListOf(dialogAct))
            } else {
                groups.last().second += dialogAct
            }
        }
        return groups.map { rewriteDialogActInGroup(it.second) }.flatten()
    }

    private fun rawMakeSingleton(qname: String) {
        val kClass = Class.forName(qname, true, chatbot!!.getLoader()).kotlin
        val ctor = kClass.primaryConstructor ?: return
        val frame = ctor.call(this) as? ISingleton
        if (frame != null) {
            frame.filler = frame.createBuilder().invoke(ParamPath(frame))
            globals[qname] = frame
        }
    }

    fun makeSingleton(qname: String) {
        if (!globals.containsKey(qname)) {
            rawMakeSingleton(qname)
        }
    }


    fun getOpenPayloadIntent(): String? {
        for (s in schedulers.reversed()) {
            val intent = (s.lastOrNull { it is FrameFiller<*> && it.frame() is IIntent && !it.frame()::class.qualifiedName!!.startsWith("io.opencui") } as? FrameFiller<*>)?.frame()
            if (intent != null) return intent::class.qualifiedName
        }
        return null
    }

    fun isOpenIntent(event: FrameEvent): Boolean {
        return schedule.firstOrNull { it is AnnotatedWrapperFiller && it.targetFiller is FrameFiller<*> && it.targetFiller.frame()::class.qualifiedName == event.fullType } != null
    }

    inline fun <reified T> getGlobal(): T? {
        val qname = T::class.qualifiedName!!
        if (qname == IUSERIDENTIFIER) {
            globals[qname] = UserIdentifier(this)
        } else {
            makeSingleton(qname)
        }
        return globals[qname] as T?
    }

    inline fun <reified T : ISingleton> getGlobalFiller(): FrameFiller<T>? {
        val qname = T::class.qualifiedName!!
        makeSingleton(qname)
        return globals[qname]?.filler as? FrameFiller<T>
    }

    fun cleanup() {
        while (schedulers.size > 1) {
            schedulers.removeLast()
        }
        mainSchedule.cleanup()
        events.clear()
        turnId = 0
        globals.clear()
        finishedIntentFiller.clear()
    }

    fun toSessionString(): String {
        val objNode = ObjectNode(JsonNodeFactory.instance)
        objNode.replace("schedulers_count", IntNode(schedulers.size))
        objNode.replace("main", mainSchedule.toObjectNode())
        return Json.encodeToString(objNode)
    }

    @kotlin.jvm.Throws(Exception::class)
    private fun readObject(ois: ObjectInputStream) {
        ois.defaultReadObject()
        chatbot = ChatbotLoader.findChatbot(botInfo)
    }

    companion object {
        val logger: Logger = LoggerFactory.getLogger(Dispatcher::class.java)
        val IUSERIDENTIFIER = IUserIdentifier::class.qualifiedName!!
        private val serialVersionUID: Long = 123
        val PACKAGE = IUSERIDENTIFIER.split(".").subList(0, 2).joinToString(".")
    }
}
