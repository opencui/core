package io.opencui.du

import io.opencui.core.*
import org.slf4j.LoggerFactory

import io.opencui.core.RuntimeConfig
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import io.opencui.serialization.*
import java.util.*

enum class DugMode {
    SKILL,
    SLOT,
    BINARY,
    SEGMENT
}


// the result from YesNoInference
enum class YesNoResult {
    Affirmative,
    Negative,
    Indifferent,
    Irrelevant
}



fun YesNoResult.toJsonAsBoolean() : String? {
    return when (this) {
        YesNoResult.Affirmative -> Json.encodeToString(true)
        YesNoResult.Negative -> Json.encodeToString(false)
        YesNoResult.Irrelevant -> Json.encodeToString(IStateTracker.DontCareLabel)
        else -> null
    }
}

//
// This is used to bridge encoder and decoder solution
//
data class TriggerDecision(
    override val utterance: String,
    override var owner: String?,  // this could be empty and can be updated by exact match.
    val evidence: List<Exemplar>) : Triggerable {

    lateinit  var duContext: DuContext

    // We use this to keep track of the nested frames
    val matchedSlots = mutableListOf<DUSlotMeta>()

    fun exactMatch(duContext: DuContext) : String? {
        // Prepare for exact match.
        evidence.map {
            it.typedExpression = it.typedExpression(duContext.duMeta!!)
        }

        val matcher = NestedMatcher(duContext)
        val candidates = evidence
        candidates.map { matcher.markMatch(it) }
        val exactMatches = candidates.find { it.exactMatch }
        return exactMatches?.ownerFrame
    }
}

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
    val client: HttpClient = HttpClient.newHttpClient()

    fun shutdown() { }

    companion object {
        val logger = LoggerFactory.getLogger(RestNluService::class.java)
    }

    data class Context(
        val bot: String
    )

    data class Request(
        val mode: DugMode,
        val utterance: String,
        val expectations: List<Map<String, String?>> = emptyList(),
        val slots: List<Map<String, String>> = emptyList(),
        val candidates: Map<String, List<String>> = emptyMap(),
        val questions: List<String> = emptyList(),
        val dialogActs: List<String> = emptyList())

    fun buildRequest(ctxt: Context, text: String, timeoutMillis: Long = 1000L): HttpRequest {
        return HttpRequest.newBuilder()
            .uri(URI.create("$url/v1/predict/${ctxt.bot}"))
            .header("Content-type", "application/json")
            .header("Accept", "application/json")
            .version(HttpClient.Version.HTTP_1_1)
            .POST(HttpRequest.BodyPublishers.ofString(text))
            .build()
    }


    // This returns skills (skills requires attention automatically even not immediately but one by one, not frames)
    fun detectTriggerables(
        ctxt: Context,
        utterance: String,
        expectations: List<Map<String, String?>> = emptyList()): List<TriggerDecision> {

        val input = Request(DugMode.SKILL, utterance, expectations)
        logger.debug("connecting to $url/v1/predict/${ctxt.bot}")
        logger.debug("utterance = $utterance and expectations = $expectations")
        val jsonRequest = Json.encodeToString(input).trimIndent()
        val request: HttpRequest = buildRequest(ctxt, jsonRequest)

        val response: HttpResponse<String> = client.send(request, HttpResponse.BodyHandlers.ofString())
        val body = response.body()
        if (response.statusCode() != 200) {
            logger.error("NLU request error: ${response.toString()}")
            return emptyList()
        }
        return Json.decodeFromString<List<TriggerDecision>>(body)
    }

    // handle all slots.
    fun fillSlots(
        ctxt: Context,
        utterance: String,
        slots: List<Map<String, String>>,
        valueCandidates: Map<String, List<String>>): Map<String, SlotValue> {
        val input = Request(DugMode.SLOT, utterance, slots = slots, candidates =  valueCandidates)
        logger.debug("connecting to $url/v1/predict/${ctxt.bot}")
        logger.debug("utterance = $utterance and expectations = $slots, entities = $valueCandidates")
        val request: HttpRequest = buildRequest(ctxt, Json.encodeToString(input))

        val response: HttpResponse<String> = client.send(request, HttpResponse.BodyHandlers.ofString())
        if (response.statusCode() != 200) {
            logger.error("NLU request error: ${response.toString()}")
            return emptyMap()
        }

        return Json.decodeFromString<Map<String, SlotValue>>(response.body())
    }

    fun yesNoInference(ctxt: Context, utterance: String, questions: List<String>): List<YesNoResult> {
        val input = Request(DugMode.BINARY, utterance, questions = questions)
        logger.debug("connecting to $url/v1/predict")
        logger.debug("utterance = $utterance and questions = $questions")
        val request: HttpRequest = buildRequest(ctxt, Json.encodeToString(input))

        val response: HttpResponse<String> = client.send(request, HttpResponse.BodyHandlers.ofString())
        if (response.statusCode() != 200) {
            logger.error("NLU request error: ${response.toString()}")
            return emptyList()
        }
        return Json.decodeFromString<List<YesNoResult>>(response.body())
    }
}


/**
 * DecoderStateTracker assumes the underlying nlu module has decoder.
 */
data class DecoderStateTracker(val duMeta: DUMeta, val forced_tag: String? = null) : IStateTracker {
    // If there are multi normalizer propose annotation on the same span, last one wins.
    val normalizers = defaultRecognizers(duMeta)

    // We can turn this on if we need to.
    val alwaysExactMatch = false
    val nluService = RestNluService(RuntimeConfig.get(RestNluService::class)?: "http://127.0.0.1:3001")

    val context : RestNluService.Context by lazy {
        val tag = if (forced_tag.isNullOrEmpty()) {
            "${duMeta.getOrg()}_${duMeta.getLabel()}_${duMeta.getLang()}_${duMeta.getBranch()}"
        } else {
            forced_tag
        }
        RestNluService.Context(tag)
    }

    val dontCareForPagedSelectable = DontCareForPagedSelectable()

    val useSlotLabel = true

    // Eventually we should use this new paradigm.
    // First, we detect triggereables this should imply skill understanding.
    // then we first handle the expectation
    // then we fill the slot.

    // For now, we assume single intent input, and we need a model before this
    // to cut multiple intent input into multiple single intent ones.

    fun buildPostProcessor(expectations: DialogExpectations): FrameEventProcessor {
        // this build the post processors
        return ChainedFrameEventProcesser(
            dontCareForPagedSelectable,        // The first is to resolve the don't care for pagedselectable.
            ComponentSkillConverter(duMeta, expectations)
        )
    }

    override fun convert(session: UserSession, putterance: String, expectations: DialogExpectations): List<FrameEvent> {
        // if it is empty, we can return immediately.
        if (putterance.trim().isEmpty()) {
            return emptyList()
        }
        // TODO(sean), eventually need to getLocale from user session, right now doing so break test.
        val utterance = putterance.lowercase(Locale.getDefault()).trim { it.isWhitespace() }

        // this layer is important so that we have a centralized place for the post process.
        val res = convertImpl(session, utterance, expectations)

        logger.info("Converted to frame events: $res")

        // get the post process done
        val postProcess = buildPostProcessor(expectations)
        return res.map { postProcess(it) }
    }

    fun buildDuContext(session: UserSession, utterance: String, expectations: DialogExpectations): DuContext {
        val ducontext = DuContext(session.userIdentifier.toString(), utterance, expectations, duMeta)
        var allNormalizers = normalizers.toMutableList()
        // Session and turn based recognizers
        if (session.sessionRecognizer != null) allNormalizers += session.sessionRecognizer!!
        if (session.turnRecognizer != null) allNormalizers += session.turnRecognizer!!

        allNormalizers.recognizeAll(
            utterance,
            ducontext.expectedEntityType(expectations),
            ducontext.entityTypeToValueInfoMap
        )

        ducontext.cleanUp(expectations)

        ducontext.updateTokens(LanguageAnalyzer.get(duMeta.getLang(), stop = false)!!.tokenize(utterance))
        return ducontext
    }

    // This layer is handling the
    fun convertImpl(session: UserSession, putterance: String, expectations: DialogExpectations): List<FrameEvent> {
        val triggerables = detectTriggerables(putterance, expectations)
        logger.debug("getting $triggerables for utterance: $putterance expectations $expectations")

        // Now we apply recognizers on every triggerable.
        for (triggerable in triggerables) {
            triggerable as TriggerDecision
            triggerable.duContext = buildDuContext(session, triggerable.utterance, expectations)
            if (triggerable.owner == null) {
                val exactOwner = triggerable.exactMatch(triggerable.duContext)

                if (exactOwner != null) {
                    logger.debug("The exactOwner $exactOwner different from guessed owner null.")
                    triggerable.owner = exactOwner
                }
            }
        }

        // Partition the triggeralbe to payload and need to be handled under context.
        val (weakTriggerables, strongTriggerables) =
            triggerables.partition { it.owner == null || isSystemFrame(it.owner!!) }

        // TODO: we might need to use exact match to make the hotfix work later.
        // We only goes into this when we have expectation, but the truth is we always have expectation.
        val results = mutableListOf<FrameEvent>()
        for (triggerable in weakTriggerables) {
            val utterance = triggerable.utterance
            val duContext = triggerable.duContext

            // Test whether it is crud, if so we hand them separately
            if (isPickValue(triggerable.owner)) {
                val expectedFrames = duContext.expectedFrames.filter {it.slot != null }
                val events = handleExpectations(triggerable, expectedFrames)
                if (!events.isNullOrEmpty()) {
                    logger.debug("getting $events for $utterance in handleExpectations")
                    // This is an opportunity for filtering the events again.
                    // if event agrees with one of expectation, and
                    results.addAll(events)
                }
            } else if (isUpdateSlot(triggerable.owner)) {
                // now we handle slot update not working yet.
                val events = handleSlotSkill(duContext, triggerable)
                if (!events.isNullOrEmpty()) {
                    logger.debug("getting $events for $utterance in handleSlotSkill")
                    results.addAll(events)
                }
            } else {
                // now we handle the no slot update cases.
                val events = handleExpectations(triggerable, duContext.expectedFrames)
                if (!events.isNullOrEmpty()) {
                    logger.debug("getting $events for $utterance in handleExpectations")
                    // This is an opportunity for filtering the events again.
                    // if event agrees with one of expectation, and
                    results.addAll(events)
                }
            }
        }

        // The rest should be payload triggerables.
        // If there are multiple triggerables, we need to handle them one by one.
        for (triggerable in strongTriggerables) {
            val utterance = triggerable.utterance
            val duContext = triggerable.duContext

            logger.debug("handling $triggerables for utterance: $utterance")
            val events = fillSlotsByFrame(duContext, triggerable.owner!!, null)
            logger.debug("getting $events for $triggerable")
            results.addAll(events)


            val frameSlots = duMeta
                .getSlotMetas(triggerable.owner!!)
                .filter { !it.isDirectFilled }
                .filter { it.triggers.isNotEmpty()  }
                .filter { !duMeta.isEntity(it.type!!) }
                .filter { duMeta.getSlotMetas(it.type!!).find {it.isHead} != null }

            for (frameSlot in frameSlots) {
                val events = fillSlotsByFrame(duContext, frameSlot.type!!, null)
                results.addAll(events)
            }

        }

        return if (results.size != 0) {
            results
        } else {
            listOf(FrameEvent.build(IStateTracker.FullIDonotKnow))
        }
    }

    fun handleSlotSkill(duContext: DuContext, triggerable: TriggerDecision) : List<FrameEvent>? {
         logger.debug("enter slot update.")
         // Potentially, we need to figure out two things, value and slot identifier. And of course
         // slot identifier need to agree with value on the types.
         val slotTypeSpanInfo = duContext.entityTypeToValueInfoMap[triggerable.owner]?.filter { !it.partialMatch }
         // Make sure there are slot type entity matches.
        return if (!slotTypeSpanInfo.isNullOrEmpty()) {
            if (slotTypeSpanInfo.size > 1) {
                // For now, we only handle
                IStateTracker.onlyHandleOneSlot()
            } else {
                // here we already know the target slot
                handleGenericSlotWithSlotType(duContext, slotTypeSpanInfo[0], triggerable)
            }
        } else {
            handleGenericSlotWithValue(duContext, triggerable)
        }
    }

    fun handleGenericSlotWithSlotType(duContext: DuContext, valueInfo: ValueInfo, triggerable: TriggerDecision) : List<FrameEvent>? {

        for (activeFrame in duContext.expectations.activeFrames) {
            // Just in case the same slot is used in different frames.
            val matchedSlotExists = isSlotMatched(duMeta, valueInfo, activeFrame.frame)
            if (!matchedSlotExists) {
                // if the claimed slot does not exist, we simply ignore.
                continue
            }

            // Here the constraint first expressed in slottype occurrances.
            val partsInQualified = valueInfo.value.toString().split(".")
            val slotName = partsInQualified.last()
            val slotsInActiveFrame = duContext.duMeta!!.getSlotMetas(activeFrame.frame)

            val targetEntitySlot = slotsInActiveFrame.find { it.label == slotName }
            if (targetEntitySlot != null) {
                val topLevelFrameType = IStateTracker.SlotUpdate
                val slotMapAft = bindTargetSlotAsGenerics(duContext, targetEntitySlot, triggerable.owner!!)
                return fillTheseSlots(duContext, slotMapAft, topLevelFrameType, null)
            } else {
                // This find the headed frame slot.
                val targetFrameType =
                    partsInQualified.subList(0, partsInQualified.size - 1).joinToString(separator = ".")
                val targetEntitySlot = duMeta.getSlotMetas(targetFrameType).find { it.label == slotName }!!
                val topLevelFrameType = IStateTracker.SlotUpdate
                val slotMapAft = bindTargetSlotAsGenerics(duContext, targetEntitySlot, triggerable.owner!!)
                return fillTheseSlots(duContext, slotMapAft, topLevelFrameType, null)
            }
        }
        return null
    }

    fun handleGenericSlotWithValue(duContext: DuContext, triggerable: TriggerDecision) : List<FrameEvent>? {
        // TODO: now we need to handle the case for: change to tomorrow
        // For now we assume there is only one generic type, and we do not know which slot is our target.
        // In case there is no guessed slot.
        val duMeta = duContext.duMeta
        for (activeFrame in duContext.expectations.activeFrames) {
            // Just in case the same slot is used in different frames.
            // now we iterate through all possible slots of this.
            for (slotMeta in duMeta!!.getSlotMetas(activeFrame.frame)) {
                // For now, we trust the recognizer
                if (!duContext.entityTypeToValueInfoMap.containsKey(slotMeta.type)) continue
                // Try this slot as the target or original slot.
                val slotMapAft = bindTargetSlotAsGenerics(duContext, slotMeta, triggerable.owner!!)
                return fillTheseSlots(duContext, slotMapAft, triggerable.owner!!, null)
            }
        }
        return null
    }

    /**
     * This is used to get new slot meta for slot skills,
     */
    fun bindTargetSlotAsGenerics(ducontext: DuContext, targetSlot: DUSlotMeta, slotSkill: String): List<DUSlotMeta> {
        // we need to make sure we include slots mentioned in the intent expression
        val slotMapBef = ducontext.duMeta!!.getSlotMetas(slotSkill)
        val slotMapTransformed = mutableListOf<DUSlotMeta>()

        // we need to rewrite the slot map to replace all the T into actual slot type.
        for (slotMeta in slotMapBef) {
            // We can not fill slot without triggers.
            if (slotMeta.isGenericTyped()) {
                // NOTE: assume the pattern for generated type is <T>
                val targetTrigger = targetSlot.triggers[1]
                val newTriggerTemplate = slotMeta.triggers.firstOrNull { it.contains(IStateTracker.ValueSymbol)}
                if (newTriggerTemplate == null) {
                    throw RuntimeException("Add a name to generic slot with <> represent the actual value")
                }
                val newTrigger =  newTriggerTemplate.replace(IStateTracker.ValueSymbol, targetTrigger)
                slotMapTransformed.add(slotMeta.typeReplaced(targetSlot.type!!, listOf(newTrigger)))
            } else {
                slotMapTransformed.add(slotMeta)
            }
        }
        return slotMapTransformed.filter { it.triggers.isNotEmpty() }
    }

    fun detectTriggerables(utterance: String, expectations: DialogExpectations): List<TriggerDecision> {
        // TODO(sean): how do we resolve the type for generic type?
        // We assume true/false or null here.
        val pcandidates = nluService.detectTriggerables(
            context,
            utterance,
            expectations.activeFrames.map { it.toDict() })

        val candidates = ChainedExampledLabelsTransformer(
            StatusTransformer(expectations)
        ).invoke(pcandidates)

        // Do we really need this?
        // First, try to exact match expressions
        // now find the intent best explain the utterance
        // First we check whether we know enough about
        if (candidates.isEmpty()) {
            logger.debug("Got no match for ${utterance}.")
        }

        // TODO: another choice is to return here and ask user to choose one interpretation.
        if (candidates.size > 1) {
            logger.debug("StateTracker.convert there is too many good matches for ${utterance}.")
        }

        // We might need to consider return multiple possibilities if there is no exact match.
        return candidates.map {it as TriggerDecision }
    }


    // When there is expectation presented.
    // For each active expectation, we do the following:
    // 1. check what type the focused slot is,
    // 2. if it is boolean/IStatus, run Yes/No inference.
    // 3. run fillSlot for the target frame.
    private fun handleExpectations(triggerable: TriggerDecision, expectedFrames: List<ExpectedFrame>): List<FrameEvent>? {
        val duContext = triggerable.duContext
        val utterance = duContext.utterance

        // Ideally, we should pay attention to
        val results = mutableListOf<FrameEvent>()
        val lowResults = mutableListOf<FrameEvent>()
        for (expectedFrame in expectedFrames) {
            val frame = expectedFrame.frame
            val slot = expectedFrame.slot

            // We need to check whether we have a binary question. For now, we only handle the top
            // binary question. We assume the first on the top.
            if (expectedFrame.isBooleanSlot(duMeta)) {
                // The expectation top need to be binary and have prompt.
                val question = expectedFrame.prompt?.firstOrNull()?.templates?.pick() ?: ""
                val hardBooleanEvents = mutableListOf<FrameEvent>()
                val softBooleanEvents = mutableListOf<FrameEvent>()
                // Hard binding to boolean value.
                val boolValue = duContext.getEntityValue(IStateTracker.KotlinBoolean)
                if (boolValue != null) {
                    val yesNoFlag = when (boolValue) {
                        "true" -> YesNoResult.Affirmative
                        "false" -> YesNoResult.Negative
                        "_DontCare" -> YesNoResult.Indifferent
                        else -> YesNoResult.Irrelevant
                    }
                    val events = handleBooleanStatus(duContext, yesNoFlag)
                    if (events != null) hardBooleanEvents.addAll(events)
                }

                val status = nluService.yesNoInference(context, utterance, listOf<String>(question))[0]
                if (status != YesNoResult.Irrelevant) {
                    // First handle the frame wrappers we had for boolean type.
                    // what happens we have good match, and these matches are related to expectations.
                    // There are at least couple different use cases.
                    // TODO(sean): should we start to pay attention to the order of the dialog expectation.
                    // Also the stack structure of dialog expectation is not used.
                    val events = handleBooleanStatus(duContext, status)
                    if (events != null) softBooleanEvents.addAll(events)
                }

                // There is no reason to have multiple yes/no from the same utterance.
                results.addAll(hardBooleanEvents.distinct())
                if (hardBooleanEvents.isEmpty()) {
                    results.addAll(softBooleanEvents.distinct())
                }
            } else if (slot != null && duMeta.getSlotType(frame, slot) == IStateTracker.KotlinString) {
                // Now w handle the string slot, this is really low priority.
                // only where owner == null
                if (triggerable.owner == null) {
                    lowResults.add(FrameEvent.build(frame, listOf(EntityEvent(duContext.utterance, slot))))
                }
            } else {
                // if there is no good match, we need to just find it using slot model.
                // try to fill slot for active frames, assuming the expected!! is the at the beginning.
                val extractedEvents = fillSlotsByFrame(duContext, expectedFrame.frame, expectedFrame.slot, true)
                logger.info("for ${expectedFrame} getting event: ${extractedEvents}")
                if (extractedEvents.isNotEmpty()) {
                    results.addAll(extractedEvents)
                }
            }
        }

        // If we have no for hasMore, but something for pagedselectable then we remove the no for hasMore
        val eventsFilter = listOf(
            HasNoMoreCleaner()
        )

        // We now return the events.
        return if (!results.isNullOrEmpty()) {
            val hasPayload = results.filter { !it.packageName!!.startsWith("io.opencui.core")}.isNullOrEmpty()
            if (hasPayload) {
               results.removeIf{ it.packageName == IStateTracker.HasMore}
            }
            filter(eventsFilter, results)
            results
        } else if (lowResults.isNotEmpty()){
            filter(eventsFilter, results)
            lowResults
        } else {
            null
        }
    }

    // This need to called if status is expected.
    private fun handleBooleanStatusImpl(boolValue: YesNoResult, valueChoices: List<String>): List<FrameEvent>? {
        // if we have extractive match.
        val frameName = when (boolValue) {
            YesNoResult.Affirmative -> valueChoices[0]
            YesNoResult.Negative -> valueChoices[1]
            YesNoResult.Indifferent -> IStateTracker.FullDontCare
            else -> null
        }
        return if (frameName != null) listOf(FrameEvent.build(frameName)) else null
    }

    private fun handleBooleanStatus(duContext: DuContext, flag: YesNoResult): List<FrameEvent>? {
        val expected = duContext.expectations.expected
        val slotType = duMeta.getSlotType(expected!!.frame, expected.slot!!)

        // First handling
        val expectations = duContext.expectations
        if (slotType == IStateTracker.ConfirmationStatus) {
            val events = handleBooleanStatusImpl(flag, IStateTracker.FullConfirmationList)
            if (events != null) return events
        }

        // b. boolgate Yes/No
        if (slotType == IStateTracker.BoolGateStatus) {
            val events = handleBooleanStatusImpl(flag, IStateTracker.FullBoolGateList)
            if (events != null) return events
        }

        // c. hasMore Yes/No
        if (slotType == IStateTracker.HasMoreStatus) {
            val events = handleBooleanStatusImpl(flag, IStateTracker.FullHasMoreList)
            if (events != null) return events
        }

        // This is used for boolean slot.
        val flagStr = flag.toJsonAsBoolean()
        // TODO(sean): make sure that expected.slot is not empty here.
        return if (expected != null && flagStr != null && expected.slot != null) {
            listOf(FrameEvent.build(expected.frame, listOf(EntityEvent(flagStr, expected.slot))))
        } else {
            return null
        }
    }


    /**
     * fillSlots is used to create entity event.
     */
    private fun fillSlotsByFrame(duContext: DuContext, topLevelFrameType: String, focusedSlot: String?, expected: Boolean=false): List<FrameEvent> {
        // we need to make sure we include slots mentioned in the intent expression
        // We only need description for slots with no direct fill.
        val slotMapBef = duMeta
            .getSlotMetas(topLevelFrameType)
            .filter { !it.isDirectFilled }

        // if not direct filled,
        val duMeta = duContext.duMeta!!
        val slotMapAft = slotMapBef
            .filter { it.triggers.isNotEmpty()  }
            .filter { duMeta.isEntity(it.type!!) }

        // For now, we only care about the single level
        if (slotMapAft.isEmpty()) {
            logger.debug("Found no slots for $topLevelFrameType")
            return if (!expected) {
                listOf(FrameEvent.build(topLevelFrameType))
            } else {
                // if this is from expected
                listOf()
            }
        }

        // TODO: figure out how to test skill slot efficiently.
        /* Hide this for now so that we can get some result
        if (slotMapBef.size != slotMapAft.size) {
            val slotsMissing = slotMapBef.filter { it.value.triggers.isEmpty() }.map {it.key }.toList()
            throw BadConfiguration("Missing triggers for slots $slotsMissing on $topLevelFrameType")
        }*/

        return fillTheseSlots(duContext, slotMapAft, topLevelFrameType, focusedSlot)
    }

    fun fillTheseSlots(
        duContext: DuContext,
        slotMetas: List<DUSlotMeta>,
        topLevelFrameType: String,
        focusedSlot: String?): List<FrameEvent> {
        // we need to make sure we include slots mentioned in the intent expression
        val nluSlotValues = mutableMapOf<String, List<String>>()

        // The question here is, do we resolve the type overlapped slot before we send to NLU?
        val duMeta = duContext.duMeta!!
        // TODO: Should this be label, or name?
        val nluSlotMetas = mutableListOf<DUSlotMeta>()
        for (slot in slotMetas) {
            val slotType = slot.type
            if (slotType != null) {
                val slotLabel = slot.label
                // For now, we only support entity level value extraction.
                val entityMeta = duMeta.getEntityMeta(slotType) ?: continue

                val valuesByType = duContext.getValuesByType(slotType)
                // if it is normalizable, and we did not recognize anything, we do not include in slot value.
                // this will leave things like name.
                if (entityMeta.normalizable && valuesByType.isNullOrEmpty()) {
                    continue
                }

                // Should we resolve the confused type where more than one slot have the same type?
                val values = valuesByType.map { duContext.utterance.substring(it.start, it.end) }
                if (!values.isNullOrEmpty()) {
                    nluSlotMetas.add(slot)
                    nluSlotValues[slotLabel] = values
                }
            }
        }

        // For now, we only focus on the equal operator, we will handle other semantics later.
        val equalSlotValues = mutableMapOf<String, MutableList<String>>()
        val nluSlots = nluSlotMetas.map { it.asMap() }.toList()
        val results = nluService.fillSlots(context, duContext.utterance, nluSlots, nluSlotValues)
        logger.debug("got $results from fillSlots for ${duContext.utterance} on $nluSlots with $nluSlotValues")

        for (entry in results.entries) {
            if(entry.value.operator == "==") {
                val values = entry.value.values.toMutableList()
                if (!values.isNullOrEmpty()) {
                    equalSlotValues[entry.key] = entry.value.values.toMutableList()
                }
            }
        }

        // Now, we need to do slot resolutions for the slots with the same type.
        val typesWithMultipleSlots = slotMetas.groupBy { it.type }.filterValues {it.size > 1}
        for (entry in typesWithMultipleSlots.entries) {
            val slotType = entry.key
            // for these type, we compute surrounding bonus
             val spannedValues = duContext.entityTypeToValueInfoMap[slotType]?.filter {
                ! it.partialMatch
            }

            if (spannedValues.isNullOrEmpty()) {
                continue
            }
            // We should have at least two slotMets here.
            for (value in spannedValues) {
                duContext.resolveSlot(value, entry.value)
            }
        }

        // Let us try to merge the evidence from recognizer.
        val entityEvents = mutableListOf<EntityEvent>()
        for (slot in nluSlotMetas) {
            val slotType = slot.type!!
            val slotName = slot.label

            // For now, we ignore the partial match.
            val valuesForType = duContext.entityTypeToValueInfoMap[slotType]?.filter { ! it.partialMatch }
            val valuesForSlot = valuesForType?.filter {slotName in it.possibleSlots || it.possibleSlots.size == 0}
            val normValueMap = valuesForSlot?.map { it.original(duContext.utterance) to it.norm() }?.toMap()

            // Regardless, we always use verified values.
            val surfaceValues = mutableSetOf<String>()
            if (!valuesForSlot.isNullOrEmpty()) {
                for (valueInfo in valuesForSlot) {
                    val value = valueInfo.original(duContext.utterance)
                    if (!normValueMap.isNullOrEmpty() && normValueMap.containsKey(value) && value !in surfaceValues) {
                        surfaceValues.add(value)
                        entityEvents.add(EntityEvent.build(slotName, value, normValueMap[value]!!, slotType))
                    }
                }
            }

            // Most of the type are normalizable, except person name for now.
            // The normalizable basically means the recognizer is dependable.
            val normalizable = duMeta!!.getEntityMeta(slotType)?.normalizable!!
            if (!normalizable && equalSlotValues.containsKey(slotName)) {
                val slotValues = equalSlotValues[slotName]!!.distinct()
                // For now, assume the operator are always equal
                for (value in slotValues) {
                    if (value in surfaceValues) continue
                    surfaceValues.add(value)
                    entityEvents.add(EntityEvent.build(slotName, value, value, slotType))
                }
            }
        }

        // Now handle partial match for focused slot, we might need to have better check.
        val focusedSlotType = if (focusedSlot.isNullOrEmpty()) null else duMeta.getSlotType(topLevelFrameType, focusedSlot)
        if (entityEvents.size == 0 && focusedSlotType != null) {
            val spannedValues = duContext.entityTypeToValueInfoMap[focusedSlotType]
            val normalizable = duMeta.getEntityMeta(focusedSlotType)?.normalizable
            // under condition where we have a unique partial match
            if (spannedValues?.size == 1 && normalizable == true) {
                val value = spannedValues[0]!!
                entityEvents.add(EntityEvent.build(focusedSlot!!, value.origValue!!, value.norm()!!, focusedSlotType))
            }
        }

        // not
        return if (!duContext.expectations.isFrameCompatible(topLevelFrameType) || entityEvents.size != 0) {
            listOf(FrameEvent.build(topLevelFrameType, entityEvents))
        } else  {
            emptyList()
        }
    }

    // given a list of frame event, add the entailed slots to the right frame event.
    override fun recycle() {
        nluService.shutdown()
    }

    companion object : ExtensionBuilder<IStateTracker> {
        val logger = LoggerFactory.getLogger(DecoderStateTracker::class.java)

        // TODO(sean): make sure entity side return this as label for DONTCARE
        override fun invoke(p1: Configuration): IStateTracker {
            TODO("Not yet implemented")
        }
    }
}
