package io.opencui.du

import com.fasterxml.jackson.annotation.JsonIgnore
import io.opencui.core.*
import org.slf4j.LoggerFactory
import java.util.*


/**
 * Dialog state tracker takes natural language user utterance, and convert that into frame event
 * based on dialog expectations that summarizes conversation history.
 *
 * For now, this functionality is separated into two levels:
 * 1. lower level nlu where context is not taking into consideration (bert).
 * 2. high level that use the output from low lever api and dialog expectation in context dependent way (kotlin).
 *
 * We will have potentially different lower level apis, for now, we assume the bert based on api
 * which is defined per document. We assume there are two models (intents and slots) for now, and
 * their apis is defined as the corresponding document.
 */

// Exemplars are used to make decisions for now.
interface Triggerable {
    val utterance: String
    var typedExpression: String
    val ownerFrame: String
    val contextFrame: String?
    val entailedSlots: List<String>
    val label: String?

    // whether it is exact match.
    var exactMatch: Boolean
    // The next two are used for potential exect match.
    var possibleExactMatch: Boolean
    var guessedSlot: DUSlotMeta?
    var score: Float

    fun clone(): Triggerable
}

/**
 * For now, we assume the most simple expectation, current frame, and current slot, and whether do-not-care
 * is turned on for target slot.
 */
data class ExpectedFrame(
    val frame: String,
    val slot: String? = null,
    @JsonIgnore val slotType: String? = null,
    @JsonIgnore val allowDontCare: Boolean? = null) {
    fun allowDontCare() : Boolean {
        // TODO(sean) remove the hard code later.
        if (frame == "io.opencui.core.PagedSelectable" && slot == "index") return true
        return allowDontCare == true
    }
}

// This is used to bridge encoder and decoder solution
data class ExampledLabel(
    override val utterance: String,
    override val ownerFrame: String,
    override val entailedSlots: List<String>,
    override val contextFrame: String? = null,
    override val label: String? = null) : Triggerable {
    override var typedExpression: String = ""
    // for now, we keep it as the last resort.
    override var exactMatch: Boolean = false
    override var possibleExactMatch: Boolean = false

    // this is used for generic typed slot by bert model.
    override var guessedSlot: DUSlotMeta? = null
    override var score: Float = 0.0f

    fun isCompatible(type: String, packageName: String?) : Boolean {
        return ownerFrame == "${packageName}.${type}"
    }

    override fun clone(): Triggerable { return this.copy() }
}

/**
 * This can be used to capture the intermediate result from understanding.
 * So that we can save some effort by avoiding repeated work.
 */
open class DuContext(
    open val session: String,
    open val utterance: String,
    open val expectations: DialogExpectations = DialogExpectations(),
    open val duMeta: DUMeta? = null)



/**
 * This is used to store the dialog expectation for the current turn.
 * activeFrames is expected to have at least one ExpectedFrame.
 * Each dialog expectation corresponds to a topic (a scheduler at UI
 * level), so we need to understand the openness of the topic so that
 * we can help to understand.
 * The order the activeFrame should be ordered by top first, the top of the scheduler
 * should show up first in the activeFrames.
 */
data class DialogExpectation(val activeFrames: List<ExpectedFrame>) {
    // This is how rest of the code current assumes.
    @JsonIgnore
    val expected: ExpectedFrame = activeFrames[0]
}

/**
 * To support multi topics, we need to one dialog expectation for each topic.
 * TODO(xiaobo): the order should be in reverse last touched order, with first one is last touched.
 */
data class DialogExpectations(val expectations: List<DialogExpectation>) {
    @JsonIgnore
    val activeFrames: List<ExpectedFrame> = expectations.reversed().map{ it.activeFrames }.flatten()
    @JsonIgnore
    val expected: ExpectedFrame? = activeFrames.firstOrNull()

    constructor(vararg expectedFrames: ExpectedFrame): this(listOf(DialogExpectation(expectedFrames.asList())))
    constructor(expectation: DialogExpectation?) : this(if (expectation != null) listOf(expectation) else emptyList())
    constructor() : this(emptyList())

    fun getFrameContext(): List<String> {
        return activeFrames.map { """{"frame_id":"${it.frame}"}""" }
    }

    fun isFrameCompatible(frameName: String) : Boolean {
        for (aframe in activeFrames) {
            if (aframe.frame.equals(frameName)) return true
        }
        return false
    }

    fun allowDontCare() : Boolean {
        for (frame in activeFrames) {
            if (frame.allowDontCare()) return true
        }
        return false
    }

    fun hasExpectation(): Boolean {
        return activeFrames.isNotEmpty()
    }
}

/**
 * The main interface for dialog understanding: converts the user utterance into structured semantic
 * representation.
 * We encourage implementation to first support uncased model, so that the same model can be used for voice
 * data without needing to truecase it.
 */
interface IStateTracker : IExtension {
    /**
     * Converts the user utterance into structured semantic representations,
     *
     * @param user dialog session, used for logging purposes.
     * @param putterance what user said in the current turn.
     * @param expectations describes the current state of dialog from chatbot side,
     * @return list of FrameEvents, structural semantic representation of what user said.
     */
    fun convert(user: String, putterance: String, expectations: DialogExpectations = DialogExpectations()): List<FrameEvent> {
        // We keep this so that all the exist test can run.
        val userSession = UserSession(user)
        return convert(userSession, putterance, expectations)
    }

    fun convert(session: UserSession, putterance: String, expectations: DialogExpectations = DialogExpectations()): List<FrameEvent>
    /**
     * Test whether a given entity event is from partial match. Mainly used for potential slot
     */
    // fun isPartialMatch(event: EntityEvent): Boolean

    /**
     * Find related entities of the same entity type given a partial matched event.
     */
    // fun findRelatedEntity(event: EntityEvent): List<String>?

    /**
     * Life cycle method, return resources allocated for this state tracker.
     */
    fun recycle()

    companion object {
        const val FullIDonotKnow = "io.opencui.core.IDonotGetIt"
        const val FullDontCare = "io.opencui.core.DontCare"
        const val SlotUpdate = "io.opencui.core.SlotUpdate"
        const val SlotType = "io.opencui.core.SlotType"
        const val DontCareLabel = "_DontCare"
        const val FullThat = "io.opencui.core.That"
        const val ThatLabel = "{'@class'='io.opencui.core.That'}"
        const val BoolGateStatus = "io.opencui.core.booleanGate.IStatus"
        val FullBoolGateList = listOf("io.opencui.core.booleanGate.Yes", "io.opencui.core.booleanGate.No")

        const val TriggerComponentSkill =  "io.opencui.core.TriggerComponentSkill"
        const val ConfirmationStatus = "io.opencui.core.confirmation.IStatus"
        val FullConfirmationList = listOf("io.opencui.core.confirmation.Yes", "io.opencui.core.confirmation.No")
        const val HasMoreStatus = "io.opencui.core.hasMore.IStatus"
        val FullHasMoreList = listOf("io.opencui.core.hasMore.Yes", "io.opencui.core.hasMore.No")
        const val KotlinBoolean = "kotlin.Boolean"
        const val SlotUpdateOriginalSlot = "originalSlot"

        const val SlotUpdateGenericType = "<T>"
        val IStatusSet = setOf(
            "io.opencui.core.confirmation.IStatus",
            "io.opencui.core.hasMore.IStatus",
            "io.opencui.core.booleanGate.IStatus")
    }
}

interface FrameEventProcessor {
    operator fun invoke(input: FrameEvent) : FrameEvent
}

class DontCareForPagedSelectable: FrameEventProcessor {
    override operator fun invoke(event: FrameEvent) : FrameEvent {
        if (event.type == "PagedSelectable" &&
            event.slots.size == 1 &&
            event.slots[0].attribute == "index" &&
            event.slots[0].value == "\"_DontCare\""
        ) {
            return buildFrameEvent(
                "io.opencui.core.PagedSelectable",
                listOf(EntityEvent(value = """"1"""", attribute = "index"))
            )
        }
        return event
    }
}


/**
 * When the current active frames contains a skill for the new skill.
 */
data class ComponentSkillConverter(
    val duMeta: DUMeta,
    val dialogExpectation: DialogExpectations) : FrameEventProcessor {

    private val expectedFrames = dialogExpectation.expectations.map { it.activeFrames }.flatten()

    override fun invoke(p1: FrameEvent): FrameEvent {
        val matched = expectedFrames.firstOrNull { expectedFrame ->
            duMeta.getSlotMetas(expectedFrame.frame).find { it.type == p1.fullType } != null
        }

        return if (matched == null) {
            return p1
        } else {
            val componentSlot = duMeta.getSlotMetas(matched.frame).firstOrNull { it.type == p1.fullType}!!
            val entityEvents = listOf(
                buildEntityEvent("compositeSkillName", matched.frame),
                buildEntityEvent("componentSkillName", componentSlot.type!!)
            )
            return buildFrameEvent(IStateTracker.TriggerComponentSkill, entityEvents)
        }
    }
}

data class ChainedFrameEventProcesser(val processers: List<FrameEventProcessor>) : FrameEventProcessor {
    constructor(vararg transformers: FrameEventProcessor): this(transformers.toList())
    override fun invoke(p1: FrameEvent): FrameEvent {
        var current = p1
        for( transform in processers) {
            current = transform(current)
        }
        return current
    }
}


/**
 * BertStateTracker assumes the underlying nlu module is bert based.
 */
interface LlmStateTracker: IStateTracker {
    val agentMeta: DUMeta

    // If there are multi normalizer propose annotation on the same span, last one wins.
    val normalizers: List<EntityRecognizer>
    val lang: String
    val dontCareForPagedSelectable: DontCareForPagedSelectable

    /**
     * Dialog expectation is used to inform DU module to be sensitive to certain information. This is important
     * as many expression can mean different things, and use expectation can make understanding a bit easy as
     * listening can be more focused.
     * Currently, there are couple different expectations:
     * 1. expecting a slot.
     * 2. expecting multi value.
     * 3. expecting confirmation.
     * 4. expecting value recommendation.
     * Of course, we can have combination of these.
     *
     * The main goal of this method is taking user utterance and convert that into frame events.
     * We follow the following process:
     * 1. find related expressions.
     * 2. use intent model to rerank the expression candidate and pick the best match and determine the frame.
     * 3. use slot model to find values for the slot for the given frame.
     * 4. generate frame events so that dialog engine can process it.
     *
     * Assumptions:
     * 1. We assume that index can be shared by different agent.
     */
    override fun convert(session: UserSession, putterance: String, expectations: DialogExpectations): List<FrameEvent> {
        logger.info("Getting $putterance under $expectations")
        // TODO(sean), eventually need to getLocale from user session, right now doing so break test.
        val utterance = putterance.lowercase(Locale.getDefault()).trim { it.isWhitespace() }
        if (utterance.isEmpty()) return listOf()

        val duContext = buildDuContext(session, putterance, expectations)
        val res = convertImpl(duContext)

        // get the post process done
        val postProcess = buildPostProcessor(expectations)
        return res.map { postProcess(it) }
    }

    fun buildPostProcessor(expectations: DialogExpectations): FrameEventProcessor {
        // this build the post processors
        return ChainedFrameEventProcesser(
            dontCareForPagedSelectable,        // The first is to resolve the don't care for pagedselectable.
            ComponentSkillConverter(agentMeta, expectations)
        )
    }

    fun buildDuContext(session: UserSession, utterance: String, expectations: DialogExpectations): DuContext

    fun convertImpl(ducontext: DuContext): List<FrameEvent>

    // This is used to recognize the triggerable skills.
    fun detectTriggerables(ducontext: DuContext): List<ExampledLabel>?

    fun handleExpectations(ducontext: DuContext): List<FrameEvent>?

    fun fillSlots(ducontext: DuContext, topLevelFrameType: String, focusedSlot: String?): List<FrameEvent>
    fun fillSlotUpdate(ducontext: DuContext, targetSlot: DUSlotMeta): List<FrameEvent>


    companion object {
        val logger = LoggerFactory.getLogger(LlmStateTracker::class.java)
        // TODO(sean): make sure entity side return this as label for DONTCARE
        const val DONTCARE = "DontCare"
    }
}


fun buildFrameEvent(
    topLevelFrame: String,
    slots: List<EntityEvent> = listOf(),
    frames: List<FrameEvent> = listOf()
): FrameEvent {
    val parts = topLevelFrame.splitToSequence(".")
    val packageName = parts.toList().subList(0, parts.count() - 1).joinToString(".", truncated = "")
    return FrameEvent(parts.last(), slots, frames, packageName)
}


fun buildEntityEvent(key: String, value: String): EntityEvent {
    return EntityEvent(value=""""$value"""", attribute=key)
}


// LlmStateTracker always try to recognize frame first, and then slot.
// We assume the output from recognizer should be taken seriously, or dependable, the quality fix should
// be inside the recognizer, not patching outside of recognizer.



// return the top k items from the collection.
fun <T : Comparable<T>> top(k: Int, collection: Iterable<T>): List<IndexedValue<T>> {
    val topList = ArrayList<IndexedValue<T>>()
    for ((index, logit) in collection.withIndex()) {
        if (topList.size < k || logit > topList.last().value) {
            topList.add(IndexedValue(index, logit))
            topList.sortByDescending { it.value }
            if (topList.size > k) {
                topList.removeAt(k)
            }
        }
    }
    return topList
}

fun <K, V> MutableMap<K, MutableList<V>>.put(key: K, value: V) {
    if (!this.containsKey(key)) {
        put(key, mutableListOf())
    }
    get(key)!!.add(value)
}


interface Resolver {
    fun resolve(ducontext: BertDuContext, before: List<ScoredDocument>): List<ScoredDocument>
}


