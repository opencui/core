package io.opencui.core.da

import io.opencui.core.*
import java.io.Serializable

// This interface represents the dialog act that bot about to take. We start from the list from schema guided
// dialog. Notice the dialog act from user side is absorbed by dialog understanding, so we do not have model
// these explicitly. These represent what bot want to express, not how they express it.

interface DialogAct: Serializable, SchemaAction {
    var templates: Templates
    override fun run(session: UserSession): ActionResult {
        val success = true
        return ActionResult(
            listOf(this),
            createLog(templates.pick()),
            success
        )
    }
}

interface SlotDialogAct: DialogAct {
    val slotName: String
    val slotType: String
    val context: List<IFrame>
}

interface FrameDialogAct: DialogAct {
    val frameType: String
}

interface DialogActRewriter : Serializable {
    var result: DialogAct
    operator fun invoke(): DialogAct = result
}

// SLOT DialogAct
data class SlotRequest(
    override val slotName: String,
    override val slotType: String,
    override val context: List<IFrame> = listOf(),
    override var templates: Templates = emptyTemplate()) : SlotDialogAct {
    // this kind of constructor is just for convenience of testcase
    constructor(slotName: String, slotType: String, templates: Templates = emptyTemplate()): this(slotName, slotType, listOf(), templates)
}

data class SlotRequestMore(
    override val slotName: String,
    override val slotType: String,
    override val context: List<IFrame> = listOf(),
    override var templates: Templates = emptyTemplate()) : SlotDialogAct {
    constructor(slotName: String, slotType: String, templates: Templates = emptyTemplate()): this(slotName, slotType, listOf(), templates)
}

data class SlotGate(
    override val slotName: String,
    override val slotType: String,
    override val context: List<IFrame> = listOf(),
    override var templates: Templates = emptyTemplate()) : SlotDialogAct {
    constructor(slotName: String, slotType: String, templates: Templates = emptyTemplate()): this(slotName, slotType, listOf(), templates)
}

enum class FailType {
    VC,
    MIN,
    MAX,
}

data class SlotNotifyFailure<T>(
    val target: T,
    override val slotName: String,
    override val slotType: String,
    val failType: FailType,
    override val context: List<IFrame> = listOf(),
    override var templates: Templates = emptyTemplate()) : SlotDialogAct {
    constructor(target: T, slotName: String, slotType: String, failType: FailType, templates: Templates = emptyTemplate(), context: List<IFrame> = listOf()): this(target, slotName, slotType, failType, context, templates)
}

data class SlotConfirm<T>(
    val target: T?,
    override val slotName: String,
    override val slotType: String,
    override val context: List<IFrame> = listOf(),
    override var templates: Templates = emptyTemplate()) : SlotDialogAct {
    constructor(target: T?, slotName: String, slotType: String, templates: Templates = emptyTemplate(), context: List<IFrame> = listOf()): this(target, slotName, slotType, context, templates)
}

data class SlotInform<T>(
    val target: T?,
    override val slotName: String,
    override val slotType: String,
    override val context: List<IFrame> = listOf(),
    override var templates: Templates = emptyTemplate()) : SlotDialogAct {
    constructor(target: T?, slotName: String, slotType: String, templates: Templates = emptyTemplate(), context: List<IFrame> = listOf()): this(target, slotName, slotType, context, templates)
}

data class SlotOffer<T>(
    val value: List<T>,
    override val slotName: String,
    override val slotType: String,
    override val context: List<IFrame> = listOf(),
    override var templates: Templates = emptyTemplate()): SlotDialogAct {
    constructor(value: List<T>, slotName: String, slotType: String, templates: Templates = emptyTemplate(), context: List<IFrame> = listOf()): this(value, slotName, slotType, context, templates)
}

data class SlotOfferSepInform<T>(
    val value: T,
    override val slotName: String,
    override val slotType: String,
    override val context: List<IFrame> = listOf(),
    override var templates: Templates = emptyTemplate()) : SlotDialogAct {
    constructor(value: T, slotName: String, slotType: String, templates: Templates = emptyTemplate(), context: List<IFrame> = listOf()): this(value, slotName, slotType, context, templates)
}

data class SlotOfferZepInform(
    override val slotName: String,
    override val slotType: String,
    override val context: List<IFrame> = listOf(),
    override var templates: Templates = emptyTemplate()) : SlotDialogAct {
    constructor(slotName: String, slotType: String, templates: Templates = emptyTemplate(), context: List<IFrame> = listOf()): this(slotName, slotType, context, templates)
}

data class SlotOfferOutlier<T>(
    val value: T,
    override val slotName: String,
    override val slotType: String,
    override val context: List<IFrame> = listOf(),
    override var templates: Templates = emptyTemplate()) : SlotDialogAct {
    constructor(value: T, slotName: String, slotType: String, templates: Templates = emptyTemplate(), context: List<IFrame> = listOf()): this(value, slotName, slotType, context, templates)
}

// FRAME DialogAct
data class FrameConfirm<T>(
    val target: T?,
    override val frameType: String,
    override var templates: Templates = emptyTemplate()) : FrameDialogAct

data class FrameInform<T>(
    val target: T?,
    override val frameType: String,
    override var templates: Templates = emptyTemplate()) : FrameDialogAct

data class FrameOffer<T>(
    val value: List<T>,
    override val frameType: String,
    override var templates: Templates = emptyTemplate()): FrameDialogAct

data class FrameOfferSepInform<T>(
    val value: T,
    override val frameType: String,
    override var templates: Templates = emptyTemplate()) : FrameDialogAct

data class FrameOfferZepInform(
    override val frameType: String,
    override var templates: Templates = emptyTemplate()) : FrameDialogAct

data class FrameOfferOutlier<T>(
    val value: T,
    override val frameType: String,
    override var templates: Templates = emptyTemplate()) : FrameDialogAct

open class UserDefinedInform<T>(
    val target: T,
    override val frameType: String,
    override var templates: Templates = emptyTemplate()) : FrameDialogAct {
    constructor(target: T, templates: Templates): this(target, target!!::class.qualifiedName!!, templates)
}

// SPECIAL Component DialogAct for Composite DialogAct
data class SlotOfferSepInformConfirm<T>(
    val target: T,
    override val slotName: String,
    override val slotType: String,
    override val context: List<IFrame> = listOf(),
    override var templates: Templates = emptyTemplate()) : SlotDialogAct

// used as a placeholder for DialogAct that is not configured yet; this kind of DialogAct should not be called at runtime
class DumbDialogAct : DialogAct {
    override var templates: Templates = TODO("Not yet implemented")
}