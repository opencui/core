package io.opencui.du

import io.opencui.core.RuntimeConfig

/**
 * To run these test locally, please start the nlu/duckling service. It requires a bit of memory,
 * so close some memory hungry application when needed.
 *
 * docker run -it --rm -p 8500:8500 -p 8501:8501 registry.cn-beijing.aliyuncs.com/ni/apps:framely__ac9152bcb17e7cc6455033701ed6e5402b266259
 * docker run -it --rm -p 8000:8000 registry.cn-beijing.aliyuncs.com/ni/apps:0616-v1
 */

open class DuTestHelper() {
    var nluHost = "127.0.0.1"
    var nluPort = 8501
    var protocol = "http"
    var ducklingAddr = "http://127.0.0.1:8000/parse"

    init {
        val ciFlag = System.getenv("ci")
        if (!ciFlag.isNullOrEmpty()) {
            protocol = "https"
            nluHost = "dug.api-us.naturali.io"
            nluPort = 443
            ducklingAddr = "http://ni-dialogflow-duckling.ni-framely.svc.cluster.local/parse"
        }
        println("[io.framely.du.StateTrackerTest] nlu addr: $nluHost:$nluPort, duckling addr: $ducklingAddr")
        RuntimeConfig.put<String>(DucklingRecognizer::class, ducklingAddr)
        RuntimeConfig.put(TfRestBertNLUModel::class, Triple(nluHost, nluPort, protocol))
    }
}
