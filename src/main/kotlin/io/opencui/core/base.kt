package io.opencui.core

import io.opencui.channel.IChannel
import io.opencui.core.da.DialogActRewriter
import io.opencui.core.user.IUserIdentifier
import io.opencui.du.*
import io.opencui.sessionmanager.ChatbotLoader
import java.io.Serializable
import kotlin.reflect.KClass
import kotlin.reflect.KMutableProperty1
import kotlin.reflect.full.*



data class BadConfiguration(val error: String) : Exception() {}


/**
 *
 * There are two aspects of type systems: one is on type sides, and another is on the
 * translation from language to schema. On type side, the difference is between is-a
 * and can_be_cast_as, on the translation side, it is rewritten to:
 * how long is your {symptom} ?
 * or make the following not frame/intent trigger
 * start from {beijing}.
 *
 * Without rewriting the training phrase, "how long did you have your {symptom}?" is not useful
 * for headache, stomachache etc.
 *
 */
interface FillBuilder : (ParamPath) -> FrameFiller<*>, Serializable

interface PolymorphicFrameGenerator: (String) -> IFrame?, Serializable

fun createFrameGenerator(session: UserSession, interfaceClassName: String) = object: PolymorphicFrameGenerator {
    override operator fun invoke(type: String): IFrame? {
        val interfaceKClass = session.findKClass(interfaceClassName) ?: return null
        val packageName = type.substringBeforeLast(".", interfaceClassName.substringBeforeLast("."))
        val simpleType = type.substringAfterLast(".")
        val frame = session.construct(packageName, simpleType, session)
        return if (interfaceKClass.isInstance(frame)) frame else null
    }
}

/**
 * One should be able to access connection, and even session. The IService contains a set of functions.
 * Service is also attached to session, just like frame.
 */
interface IEntity : Serializable{
    var value: String
    var origValue: String?
}

/**
 * For value disambiguation, we need to expose some information for the generic implementation
 * for the slot that we bind dynamically.
 * Builder can also do slotNames.random() and typeNames.random().
 */
interface IFrame : Serializable {
    var session: UserSession?

    fun getUserIdentifier(): IUserIdentifier? {
        return session
    }

    fun annotations(path: String): List<Annotation> = listOf()

    fun createBuilder(): FillBuilder

    // slot "this" is a special slot which indicates searching for frame confirmation
    fun searchConfirmation(path: String): IFrame? {
        return null
    }

    fun searchStateUpdateByEvent(event: String): IFrameBuilder? {
        return null
    }
}

inline fun <reified T : Annotation> IFrame.findAll(path: String): List<T> =
    annotations(path).filter { it is T && it.switch() }.map { it as T }

interface IIntent : IFrame {
    // TODO(xiaobo, xiaoyun): all the filling related property should be in filler instead of frame, ideally.
    fun searchResponse(): Action? {
        return null
    }
}

// This is used to make sure
interface IBotMode
interface IKernelIntent: IBotMode, IIntent


@Throws(NoSuchMethodException::class)
fun invokeMethodByReflection(receiver: Any, funName: String, vararg params: Any?): Any? {
    val kClass = receiver::class
    val method = kClass.memberFunctions.firstOrNull { it.name == funName && it.parameters.size == params.size + 1 } ?: throw NoSuchMethodException("no such method : $funName") // param of method include receiver
    return method.call(receiver, *params)
}

@Throws(NoSuchPropertyException::class)
fun getPropertyValueByReflection(receiver: Any, propertyName: String): Any? {
    val kClass = receiver::class
    val property = kClass.members.firstOrNull { it.name == propertyName } as? KMutableProperty1<Any, *> ?: throw NoSuchPropertyException()
    return property.get(receiver)
}


// This is used to pass the runtime configure, information needed for all agents.
object RuntimeConfig {
    val configures = mutableMapOf<KClass<*>, Any>()
    inline fun <reified T: Any> put(key: KClass<*>, value: T) {
        if (configures.containsKey(key)) {
            Dispatcher.logger.info("$key is configured twice, was ${configures[key]} now $value")
        }
        configures[key] = value
    }

    inline fun <reified T: Any> update(key: KClass<*>, value: T) {
        configures[key] = value
    }

    inline fun <reified T: Any> get(key: KClass<*>) : T? {
        return configures[key] as T?
    }
}


data class RoutingInfo(val id: String, val intentsDesc: List<String>)


interface BotInfo: Serializable {
    val fullName: String
    val lang: String
    val branch: String
}

fun master(lang: String = "*") : BotInfo {
    return object : BotInfo {
        override val fullName =  Dispatcher.botPrefix!!
        override val lang = lang
        override val branch = "master" }
}
fun botInfo(fullName: String) : BotInfo {
    return object : BotInfo {
        override val fullName =  fullName
        override val lang = "*"
        override val branch = "master" }
}
fun botInfo(org: String, bot: String) : BotInfo {
    return object : BotInfo {
        override val fullName =  "${org}.${bot}"
        override val lang = "*"
        override val branch = "master" }
}
fun botInfo(fullName: String, lang: String, branch: String) : BotInfo {
    return object : BotInfo {
        override val fullName =  fullName
        override val lang = lang
        override val branch = branch }
}
fun botInfo(org: String, bot: String, lang: String, branch: String) : BotInfo {
    return object : BotInfo {
        override val fullName =  "${org}.${bot}"
        override val lang = lang
        override val branch = branch }
}

interface Component {
    val orgName: String
    val agentName: String
    val agentBranch: String
    val agentVersion: String
    val agentLang: String
    val timezone: String
}

/**
 * The chatbot implementation will be used to hold the data/filler together.
 * TODO: why this is NOT implement AgentMeta?
 */
abstract class IChatbot : Component {
    abstract val duMeta: DUMeta

    override val orgName: String = duMeta.getOrg()
    override val agentName: String = duMeta.getLabel()
    override val agentBranch: String = duMeta.getBranch()
    override val agentVersion: String = duMeta.getVersion()
    override val agentLang: String = duMeta.getLang()
    override val timezone: String = duMeta.getTimezone()

    // Do we have support connected behind bot?
    abstract val stateTracker: IStateTracker

    // This is used to host extension managers.
    val extensions: ExtensionManager = ExtensionManager()

    // This is used for hosting dialog act rewrite rule.
    abstract val rewriteRules: List<KClass<out DialogActRewriter>>

    // This is designed for routing conversation to right department when needed.
    abstract val routing: Map<String, RoutingInfo>

    fun getChannel(label: String) : IChannel? {
        return extensions.get<IChannel>(label)
    }

    inline fun <reified T:IExtension> getExtension(label: String): T? {
        return extensions.get(label)
    }

    inline fun <reified T : IExtension> getExtension(): T? {
        val labels = extensions.labelsByInterface[T::class] ?: emptyList()
        if (labels.isEmpty()) return null
        return extensions.get(labels[0])
    }

    fun getConfiguration(label: String): Configuration? {
        return Configuration.get(label)
    }

    fun createUserSession(channelType: String, user: String, channelLabel: String?): UserSession{
        return UserSession(user, channelType, channelLabel, chatbot = this)
    }

    open fun recycle() {
    }

    fun getLoader() : ClassLoader = ChatbotLoader.findClassLoader(botInfo(orgName, agentName, agentLang, agentBranch))

    operator fun invoke(p1: String, session: UserSession, packageName: String? = null): FillBuilder? {
        // hardcode for clean session
        val revisedPackageName = packageName ?: this.javaClass.packageName
        return session.construct(revisedPackageName, p1, session)?.createBuilder()
    }

    companion object {
        val ExpressionPath = "expression.json"
        val EntityPath = "entity.json"
        val EntityMetaPath = "entity_meta.json"
        val SlotMetaPath = "slot_meta.json"
        val AliasPath = "alias.json"

        fun parseEntityToMapByNT(entity: String, entries: String): Map<String, List<String>> {
            println("processing $entity with $entries")
            // content are encoded by newline and tab.
            val lines = entries.split("\n").map{ it.split("\t") }

            val nempties = lines.count{ it[0] == ""}
            val nnemptties = lines.count{ it[0] != ""}

            assert (nempties == 0 || nnemptties == 0)

            val map: MutableMap<String, List<String>> = mutableMapOf()
            for (l in lines.withIndex()) {
                val norm =  if (nnemptties != 0) l.value[0] else "${entity}.${l.index}"
                map[norm] = l.value.subList(1, l.value.size)
            }
            return map
        }

        fun loadDUMetaDsl(langPack: LangPack, classLoader: ClassLoader, org: String, agent: String, lang: String, branch: String, version: String, timezone: String = "america/los_angeles"): DUMeta {
            return object : DslDUMeta() {
                override val entityTypes = langPack.entityTypes
                override val slotMetaMap = langPack.frameSlotMetas
                override val aliasMap = langPack.typeAlias
                override val skills = langPack.skills

                init {
                    val surroundings = extractSlotSurroundingWords(expressionsByFrame, LanguageAnalyzer.get(lang, stop=false)!!)
                    for ((frame, slots) in slotMetaMap) {
                        for (slot in slots) {
                            slot.prefixes = surroundings.first["$frame:${slot.label}"]
                            slot.suffixes = surroundings.second["$frame:${slot.label}"]
                        }
                    }
                }

                override fun getSubFrames(fullyQualifiedType: String): List<String> { return subtypes[fullyQualifiedType] ?: emptyList() }

                override fun getOrg(): String = org
                override fun getLang(): String = lang
                override fun getLabel(): String = agent
                override fun getVersion(): String = version
                override fun getBranch(): String = branch
                override fun getTimezone(): String = timezone

                override fun getEntityInstances(name: String): Map<String, List<String>> {
                    return langPack.entityTypes[name]!!.entities
                }

                override val expressionsByFrame: Map<String, List<Exemplar>>
                    get() = langPack.frames
            }
        }
    }
}
