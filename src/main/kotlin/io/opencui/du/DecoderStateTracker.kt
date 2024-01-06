package io.opencui.du

import io.opencui.core.*
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.runBlocking
import org.slf4j.LoggerFactory
import kotlinx.coroutines.async
import java.util.*
import kotlin.collections.ArrayList
import kotlin.collections.HashMap
import kotlin.math.max
import kotlin.math.min

import io.opencui.core.RuntimeConfig
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import io.opencui.serialization.*
import java.time.Duration


enum class DugMode {
    SKILL,
    SLOT,
    BINARY,
    SEGMENT
}

enum class BinaryResult {
    TRUE,
    FALSE,
    DONTCARE,
    IRRELEVANT
}

data class SlotValue(val operator: String, val values: List<String>)

/**
 * For RAG based solution, there are two different stage, build prompt, and then use model to score using
 * the generated prompt. It is possible that we have two different service, one for prompt (which is agent
 * dependent), and one for low level NLU, which can be shared by multiple agents.
 *
 * The encoder model works the same way, except the retrieval is done in the Kotlin side.
 */

// Most likely, the agent dependent nlu and fine-tuned decoder are co-located, so it is better to
// hide that from user.

data class RestNluService(val url: String) {
    data class TfRestPayload(val utterance: String, val probes: List<String>)
    data class TfRestRequest(val signature_name: String, val inputs: TfRestPayload)
    val config: Triple<String, Int, String> = RuntimeConfig.get(TfRestBertNLUModel::class)
    val client: HttpClient = HttpClient.newHttpClient()
    // val url: String = "${config.third}://${config.first}:${config.second}"
    val timeout: Long = 10000


    fun parse(modelName: String, signatureName: String, utterance: String, probes: List<String>) : JsonObject? {
        val payload = TfRestPayload(utterance, probes)
        val input = TfRestRequest(signatureName, payload)
        logger.debug("connecting to $url/v1/models/${modelName}:predict")
        logger.debug("utterance = $utterance and probes = $probes")
        val request: HttpRequest = HttpRequest.newBuilder()
            .POST(HttpRequest.BodyPublishers.ofString(Json.encodeToString(input)))
            .uri(URI.create("$url/v1/models/${modelName}:predict"))
            .timeout(Duration.ofMillis(timeout))
            .build()

        val response: HttpResponse<String> = client.send(request, HttpResponse.BodyHandlers.ofString())
        return if (response.statusCode() == 200) {
            val body = response.body()
            Json.parseToJsonElement(body).get(TfBertNLUModel.outputs) as JsonObject
        } else {
            // We should not be here.
            logger.error("NLU request error: ${response.toString()}")
            null
        }
    }

    fun shutdown() { }

    fun predictIntent(lang: String, utterance: String, exemplars: List<String>): IntentModelResult? {
        val outputs = parse("${lang}_intent", "intent", utterance, exemplars)!!
        val classLogits = Json.decodeFromJsonElement<List<List<Float>>>(outputs.get(TfBertNLUModel.unifiedLogitsStr)).flatten()
        return IntentModelResult(classLogits)
    }

    fun predictSlot(lang: String, utterance: String, probes: List<String>): UnifiedModelResult {
        val outputs = parse("${lang}_slot", "slot", utterance, probes)!!
        val segments = Json.decodeFromJsonElement<List<List<String>>>(outputs.get(TfBertNLUModel.segmentsStr)).flatten()
        val startLogitss = Json.decodeFromJsonElement<List<List<Float>>>(outputs.get(TfBertNLUModel.startLogitsStr))
        val endLogitss = Json.decodeFromJsonElement<List<List<Float>>>(outputs.get(TfBertNLUModel.endLogitsStr))
        val classLogits = Json.decodeFromJsonElement<List<List<Float>>>(outputs.get(TfBertNLUModel.unifiedLogitsStr)).flatten()
        val segStarts = Json.decodeFromJsonElement<List<List<Long>>>(outputs.get(TfBertNLUModel.segStartStr)).flatten()
        val segEnds = Json.decodeFromJsonElement<List<List<Long>>>(outputs.get(TfBertNLUModel.segEndStr)).flatten()
        return UnifiedModelResult(segments, classLogits, startLogitss, endLogitss, segStarts, segEnds)
    }

    companion object {
        val logger = LoggerFactory.getLogger(TfRestBertNLUModel::class.java)
    }

    fun detectSkills(utterance: String, expectations: DialogExpectations): List<ExampledLabel> {
        TODO("Not yet implemented")
    }

    // handle all slots.
    fun fillSlots(utterance: String, slots: Map<String, DUSlotMeta>, entities: Map<String, List<String>>): Map<String, SlotValue> {
        TODO("Not yet implemented")
    }

    fun yesNoInference(utterance: String, questions: String): BinaryResult {
        TODO("Not yet implemented")
    }
}



/**
 * DecoderStateTracker assumes the underlying nlu module has decoder.
 */
data class DecoderStateTracker(
    override val agentMeta: DUMeta,
    val nluServiceUrl: String,
) : LlmStateTracker {
    // If there are multi normalizer propose annotation on the same span, last one wins.
    override val normalizers = defaultRecognizers(agentMeta)
    val nluService = RestNluService(nluServiceUrl)

    override val lang = agentMeta.getLang().lowercase(Locale.getDefault())
    override val dontCareForPagedSelectable = DontCareForPagedSelectable()

    // Eventually we should use this new paradigm.
    // First, we detect triggereables this should imply skill understanding.
    // then we first handle the expectation
    // then we fill the slot.

    // For now, we assume single intent input, and we need a model before this
    // to cut multiple intent input into multiple single intent ones.
    override fun detectTriggerables(ducontext: DuContext): List<ExampledLabel>? {
        // TODO(sean): how do we resolve the type for generic type?

        // recognized entities in utterance
        val emap = ducontext.entityTypeToSpanInfoMap
        val utterance = ducontext.utterance
        val expectations = ducontext.expectations

        // We assume true/false or null here.
        val pcandidates = nluService.detectSkills(utterance, expectations)
        val candidates = ChainedExampledLabelsTransformer(
            DontCareTransformer(ducontext.expectations),
            StatusTransformer(ducontext.expectations)
        ).invoke(pcandidates)


        // Do we really need this?
        // First, try to exact match expressions
        val matcher = NestedMatcher(ducontext)
        candidates.map { matcher.markMatch(it) }
        val exactMatches = candidates.filter { it.exactMatch }
        if (exactMatches.isNotEmpty()) {
            return exactMatches.map{ it as ExampledLabel }
        }

        // If we have potential exact match, we use that as well.
        val possibleExactMatches = candidates.filter { it.possibleExactMatch }
        if (possibleExactMatches.isNotEmpty()) {
            return possibleExactMatches.map{ it as ExampledLabel }
        }

        // now find the intent best explain the utterance
        // First we check whether we know enough about
        if (candidates.isEmpty()) {
            logger.debug("Got no match for ${utterance}.")
            return null
        }

        // TODO: another choice is to return here and ask user to choose one interpretation.
        if (candidates.size > 1) {
            logger.debug("StateTracker.convert there is too many good matches for ${utterance}.")
        }

        // We might need to consider return multiple possibilities if there is no exact match.
        return candidates.map {it as ExampledLabel }
    }

    // When there is expectation presented.
    // For each active expectation, we do the following:
    // 1. check what type the focused slot is,
    // 2. if it is boolean/IStatus, run Yes/No inference.
    // 3. run fillSlot for the target frame.

    override fun handleExpectations(ducontext: DuContext): List<FrameEvent>? {
        val candidates = ducontext.exemplars
        val expectations = ducontext.expectations
        if (candidates?.size == 1
            && !agentMeta.isSystemFrame(candidates[0].ownerFrame)
            && !expectations.isFrameCompatible(candidates[0].ownerFrame)) return null

        logger.debug(
            "${ducontext.bestCandidate} enter convertWithExpection ${expectations.isFrameCompatible(IStateTracker.ConfirmationStatus)} and ${
                ducontext.matchedIn(
                    IStateTracker.FullConfirmationList
                )
            }"
        )



        // what happens we have good match, and these matches are related to expectations.
        // There are at least couple different use cases.
        // TODO(sean): should we start to pay attention to the order of the dialog expectation.
        // Also the stack structure of dialog expectation is not used.
        // a. confirm Yes/No
        if (expectations.isFrameCompatible(IStateTracker.ConfirmationStatus)) {
            val events = handleExpectedBoolean(ducontext, IStateTracker.FullConfirmationList)
            if (events != null) return events
        }

        // b. boolgate Yes/No
        if (expectations.isFrameCompatible(IStateTracker.BoolGateStatus)) {
            val events = handleExpectedBoolean(ducontext, IStateTracker.FullBoolGateList)
            if (events != null) return events
        }

        // c. hasMore Yes/No
        if (expectations.isFrameCompatible(IStateTracker.HasMoreStatus)) {
            val events = handleExpectedBoolean(ducontext, IStateTracker.FullHasMoreList)
            if (events != null) return events
        }

        // d. match Dontcare expression abstractively
        if (ducontext.bestCandidate?.ownerFrame == IStateTracker.FullDontCare && expectations.hasExpectation()) {
            logger.debug("enter dontcare check.")
            // There are two cases where we have DontCare:
            // the best candidate has no context or its context matches expectations
            val bestCandidate = ducontext.bestCandidate!!
            // we need to go through all the expectation
            for (expected in ducontext.expectations.activeFrames) {
                if (!expected.allowDontCare()) continue
                if (bestCandidate.contextFrame == null || bestCandidate.contextFrame == expected.frame) {
                    val slotType = agentMeta.getSlotType(expected.frame, expected.slot!!)
                    // TODO: handle the frame slot case.
                    if (agentMeta.isEntity(slotType)) {
                        return listOf(
                            buildFrameEvent(
                                expected.frame,
                                listOf(EntityEvent("\"_DontCare\"", expected.slot))
                            )
                        )
                    }
                }
            }
        }

        // Now we need to figure out what happens for slotupdate.
        if (ducontext.bestCandidate?.ownerFrame == IStateTracker.SlotUpdate && expectations.hasExpectation()) {
            logger.debug("enter slot update.")
            // We need to figure out which slot user are interested in first.
            val slotTypeSpanInfo = ducontext.entityTypeToSpanInfoMap[IStateTracker.SlotType]
            // Make sure there are slot type entity matches.
            if (slotTypeSpanInfo != null) {
                // We assume the expectation is stack, with most recent frames in the end
                for (activeFrame in ducontext.expectations.activeFrames) {
                    val matchedSlotList = slotTypeSpanInfo.filter { isSlotMatched(it, activeFrame.frame) }
                    if (matchedSlotList.isEmpty()) {
                        continue
                    }

                    // Dedup first.
                    val matchedSlots = matchedSlotList.groupBy { it.value.toString() }
                    if (matchedSlots.size > 1) {
                        throw RuntimeException("Can not mapping two different slot yet")
                    }

                    // check if the current frame has the slot we cared about and go with that.
                    val spanInfo = matchedSlotList[0]
                    val partsInQualified = spanInfo.value.toString().split(".")
                    val slotName = partsInQualified.last()
                    val slotsInActiveFrame = agentMeta.getSlotMetas(activeFrame.frame)

                    val targetEntitySlot = slotsInActiveFrame.find { it.label == slotName }
                    if (targetEntitySlot != null) {
                        return fillSlotUpdate(ducontext, targetEntitySlot)
                    } else {
                        // This find the headed frame slot.
                        val targetFrameType =
                            partsInQualified.subList(0, partsInQualified.size - 1).joinToString(separator = ".")
                        val targetEntitySlot = agentMeta.getSlotMetas(targetFrameType).find { it.label == slotName }!!
                        return fillSlotUpdate(ducontext, targetEntitySlot)
                    }
                }
            } else {
                // TODO: now we need to handle the case for: change to tomorrow
                // For now we assume there is only one generic type.
                val bestCandidate = ducontext.bestCandidate!!
                val targetSlot = bestCandidate.guessedSlot!!
                return fillSlotUpdate(ducontext, targetSlot)
            }
        }

        // if there is no good match, we need to just find it using slot model.
        val extractedEvents0 = fillSlots(ducontext, expectations.expected!!.frame, expectations.expected.slot)
        if (extractedEvents0.isNotEmpty()) {
            return extractedEvents0
        }

        // try to fill slot for active frames
        for (activeFrame in expectations.activeFrames) {
            val extractedEvents = fillSlots(ducontext, activeFrame.frame, activeFrame.slot)
            logger.info("for ${activeFrame} getting event: ${extractedEvents}")
            if (extractedEvents.isNotEmpty()) {
                return extractedEvents
            }
        }

        // TODO: when we have better intent model, we can move this the end of the convert.
        if (expectations.expected.slot != null) {
            // First step, handle the basic string case.
            val frame = expectations.expected.frame
            val slot = expectations.expected.slot
            if (agentMeta.getSlotType(frame, slot).equals("kotlin.String")) {
                return listOf(
                    buildFrameEvent(
                        expectations.expected.frame,
                        listOf(EntityEvent(ducontext.utterance, slot))
                    )
                )
            }
        }
        return null
    }

    private fun isSlotMatched(spanInfo: SpanInfo, activeFrame: String): Boolean {
        val spanTargetSlot = spanInfo.value.toString()
        val parts = spanTargetSlot.split(".")
        val spanTargetFrame = parts.subList(0, parts.size - 1).joinToString(separator = ".")
        val slotName = parts.last()
        val slotMeta = agentMeta.getSlotMeta(spanTargetFrame, slotName)!!
        if (spanTargetSlot.startsWith(activeFrame) && agentMeta.isEntity(slotMeta.type!!)) return true

        val spanTargetFrameHasHead = agentMeta.getSlotMetas(spanTargetFrame).any { it.isHead }
        // now we need to figure out whether active Frame as a frame slot of this time.
        val matchedFrameSlots = agentMeta.getSlotMetas(activeFrame).filter { it.type == spanTargetFrame }
        return spanTargetFrameHasHead && matchedFrameSlots.size == 1
    }


    // This need to called if status is expected.
    private fun handleExpectedBoolean(ducontext: DuContext, valueChoices: List<String>): List<FrameEvent>? {
        if (ducontext.matchedIn(valueChoices)) {
            return listOf(buildFrameEvent(ducontext.bestCandidate?.label!!))
        }
        // if we have extractive match.
        val boolValue = ducontext.getEntityValue(IStateTracker.KotlinBoolean)
        if (boolValue != null) {
            val frameName = when (boolValue) {
                "true" -> valueChoices[0]
                "false" -> valueChoices[1]
                else -> null
            }
            if (frameName != null) return listOf(buildFrameEvent(frameName))
        }
        return null
    }

    /**
     * fillSlots is used to create entity event.
     */
    override fun fillSlots(ducontext: DuContext, topLevelFrameType: String, focusedSlot: String?): List<FrameEvent> {
        // we need to make sure we include slots mentioned in the intent expression
        val slotMap = agentMeta
            .getNestedSlotMetas(topLevelFrameType, emptyList())
            .filter { it.value.triggers.isNotEmpty() }
        return fillSlots(slotMap, ducontext, topLevelFrameType, focusedSlot)
    }

    override fun fillSlotUpdate(ducontext: DuContext, targetSlot: DUSlotMeta): List<FrameEvent> {
        TODO("Not yet implemented")
    }

    private fun fillSlots(
        slotMap: Map<String, DUSlotMeta>,
        ducontext: DuContext,
        topLevelFrameType: String,
        focusedSlot: String?
    ): List<FrameEvent> {
        // we need to make sure we include slots mentioned in the intent expression
        val valuesFound = mapOf<String, List<String>>()
        val result = nluService.fillSlots(ducontext.utterance, slotMap, valuesFound)

        return  listOf(buildFrameEvent(topLevelFrameType))
    }


    // given a list of frame event, add the entailed slots to the right frame event.

    override fun recycle() {
        nluService.shutdown()
    }

    companion object : ExtensionBuilder<IStateTracker> {
        val logger = LoggerFactory.getLogger(BertStateTracker::class.java)

        // TODO(sean): make sure entity side return this as label for DONTCARE
        const val DONTCARE = "DontCare"
        override fun invoke(p1: Configuration): IStateTracker {
            TODO("Not yet implemented")
        }
    }
}

