package io.opencui.core

import com.fasterxml.jackson.annotation.JsonIgnore
import com.fasterxml.jackson.annotation.JsonInclude
import com.fasterxml.jackson.databind.node.ArrayNode
import com.fasterxml.jackson.databind.node.TextNode
import com.fasterxml.jackson.databind.node.ValueNode
import io.opencui.serialization.*
import java.io.Serializable


// This is used for reference.
// Reference can be used to find the referent in the context.
interface Reference
data class That(val slot: String? = null): Reference {}
data class ListThat(val index: Int, val slot: String? = null): Reference {}
data class ContentThat<T>(val value: T): Reference {}


/**
 * This is used to keep the input from user. But it can be produced by some rules.
 * In general, builder can not define the label that starts with _, or they are {} wrapped.
 * NOTE:
 * value == "" means that "does not care"
 * value == "{}" means that we need to find actual value from contextual referent, there are different references.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
data class EntityEvent(
    val value: String,
    val attribute: String): Serializable {
    var type: String? = null
    @JsonIgnore
    var isUsed: Boolean = false
    var origValue: String? = null

    var isLeaf: Boolean = true

    var semantic : CompanionType = CompanionType.AND

    fun toCompanion(companionType: CompanionType) : EntityEvent {
        return EntityEvent(value, "${attribute}_", type).apply { semantic = companionType }
    }

    fun toOriginal(companionType: CompanionType) : EntityEvent {
        return EntityEvent(value, attribute, type).apply { semantic = companionType }
    }

    fun toLongForm() : String {
        return """EntityEvent(value=$value, attribute=$attribute, isLeaf=$isLeaf, type=$type)"""
    }

    constructor(value: String, attribute: String, type: String?) : this(value, attribute) {
        this.type = type
    }

    // TODO(sean) what is this used for?
    @JsonIgnore
    val decorativeAnnotations: MutableList<Annotation> = mutableListOf()

    companion object {
        fun build(key: String, value: String): EntityEvent {
            return EntityEvent(value=""""$value"""", attribute=key)
        }
        fun build(key: String, porigValue: String, value: String, ptype: String): EntityEvent {
            return EntityEvent(value=value, attribute=key).apply{
                type = ptype
                origValue = porigValue
            }
        }
    }
}

enum class EventSource {
    USER,   // From user query.
    API,  // From api.
    UNKNOWN
}

/**
 * This is used for specify proposed template match, each contains one trigger, and
 * multiple slot filling.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
data class FrameEvent(
        var type: String,
        val slots: List<EntityEvent> = emptyList(),
        val frames: List<FrameEvent> = emptyList(),
        var packageName: String? = null): Serializable {
    var attribute: String? = null
    var query: String? = null

    fun toCompanion(companionType: CompanionType) : FrameEvent {
        return FrameEvent(type, slots.map { it.toCompanion(companionType) }, frames, packageName)
    }

    val qualifiedName = "$packageName.$type"

    @JsonIgnore
    var triggered: Boolean = false

    @JsonIgnore
    var inferredFrom: Boolean = false

    @JsonIgnore
    var refocused: Boolean = false

    @JsonIgnore
    var typeUsed: Boolean = false


    @get:JsonIgnore
    val isUsed: Boolean
        get() = typeUsed || slots.firstOrNull { it.isUsed } != null || frames.firstOrNull { it.isUsed } != null

    @get:JsonIgnore
    val consumed : Boolean
        get() = slots.firstOrNull { !it.isUsed } == null && frames.firstOrNull { !it.usedUp } == null

    @get:JsonIgnore
    val usedUp: Boolean
        get() = ((slots.isNotEmpty() || frames.isNotEmpty()) && consumed) || (slots.isEmpty() && frames.isEmpty() && typeUsed)

    @get:JsonIgnore
    val activeEntitySlots: List<EntityEvent>
        get() = slots.filter {!it.isUsed }

    @get:JsonIgnore
    val activeFrameSlots: List<FrameEvent>
        get() = frames.filter {!it.isUsed }

    @get:JsonIgnore
    val fullType: String
        get() = "${packageName}.$type"

    @JsonIgnore
    val triggerParameters: MutableList<Any?> = mutableListOf()
    @JsonIgnore
    var slotAssignments: MutableMap<String, ()->Any?> = mutableMapOf()

    @JsonIgnore
    var turnId : Int = -1

    fun updateTurnId(pturnId: Int) {
        turnId = pturnId
    }

    var source: EventSource = EventSource.UNKNOWN

    companion object {
        // TODO(xiaobo): is this frameName simple or qualified?
        fun fromJson(frameName: String, jsonElement: JsonElement): FrameEvent {
            val slotEvents = mutableListOf<EntityEvent>()
            if (jsonElement is JsonObject) {
                for ((k, v) in jsonElement.fields()) {
                    if (k == "@class") continue
                    if (v is ValueNode && v !is JsonNull) {
                        slotEvents += EntityEvent(v.toString(), k)
                    } else if (v is ArrayNode) {
                        assert(v.size() == 2)
                        slotEvents += EntityEvent((v[1] as ValueNode).toString(), k).apply {
                            type = (v[0] as TextNode).textValue()
                        }
                    }
                }
            }
            return FrameEvent(frameName, slotEvents)
        }

        fun build(topLevelFrame: String,
            slots: List<EntityEvent> = listOf(),
            frames: List<FrameEvent> = listOf()
        ): FrameEvent {
            val parts = topLevelFrame.splitToSequence(".")
            val packageName = parts.toList().subList(0, parts.count() - 1).joinToString(".", truncated = "")
            return FrameEvent(parts.last(), slots, frames, packageName)
        }
    }
}

@JsonInclude(JsonInclude.Include.NON_NULL)
data class ParsedQuery(
    val query: String,
    var frames: List<FrameEvent>) {
    init {
        for (event in frames) {
            event.query = query
        }
    }
}