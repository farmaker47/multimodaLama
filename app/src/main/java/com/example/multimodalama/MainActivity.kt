package com.example.multimodalama

import android.os.Bundle
import android.util.Base64
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.example.multimodalama.ui.theme.MultimodaLamaTheme
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.WritableMap
import com.google.gson.Gson
import com.rnllama.RNLlama
import java.io.File
import kotlin.math.roundToInt

class MainActivity : ComponentActivity() {

    private lateinit var rnLlm: RNLlama
    private var isLlmReady by mutableStateOf(false)
    private var isAudioReady by mutableStateOf(false)
    private var isCompleting by mutableStateOf(false)
    private var llmStatus by mutableStateOf("Initializing...")
    private var completionResult by mutableStateOf("")
    private var isMultimodalReady by mutableStateOf(false)
    private var tokensPerSecond by mutableIntStateOf(0)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // The RNLlama class needs a ReactApplicationContext. We can create one from our app context.
        val reactContext = ReactApplicationContext(applicationContext)
        rnLlm = RNLlama(reactContext)

        // Use either vision OR no vision models.
        // initializeLlmContext()
        // initializeAudio()
        initializeVisionLlmContext()

        setContent {
            MultimodaLamaTheme {
                LlmDemoScreen(
                    status = llmStatus,
                    tokensPerSecond = tokensPerSecond,
                    result = completionResult,
                    isReady = isLlmReady,
                    isMultimodalReady = isMultimodalReady,
                    isCompleting = isCompleting,
                    onStartChatCompletion = { isStreaming ->
                        startAnyCompletion(isStreaming)
                        tokensPerSecond = 0
                    },
                    onStartVisionCompletion = {
                        startVisionCompletion()
                        tokensPerSecond = 0
                    }
                )
            }
        }
    }

    private fun initializeLlmContext() {
        // Multimodals:
        // https://huggingface.co/collections/ggml-org/multimodal-ggufs-68244e01ff1f39e5bebeeedc
        // https://github.com/ggml-org/llama.cpp/tree/master/tools/mtmd#how-to-obtain-mmproj
        // https://huggingface.co/Mungert/SmolDocling-256M-preview-GGUF/tree/main
        // Important: Download the file.gguf AND the mmproj.gguf

        val params: WritableMap = Arguments.createMap().apply {
            putString(
                "model",
                "/data/local/tmp/google_gemma-3-1b-it-qat-Q8_0.gguf"
            )
            putInt("n_ctx", 2048) // Context size
            putInt("n_gpu_layers", 0) // GPU layers (0 for CPU on Android for now)
            putBoolean("embedding", false)
        }

        val promise = object : Promise {
            override fun reject(code: String, message: String?) {

            }

            override fun reject(code: String, throwable: Throwable?) {
            }

            override fun reject(code: String, message: String?, throwable: Throwable?) {
                Log.e(
                    "Llm init",
                    "Failed to initialize Llm context: $code - $message",
                    throwable
                )
                runOnUiThread {
                    llmStatus = "Error: Failed to load Llm context.\n${message}"
                }
            }

            override fun reject(throwable: Throwable) {
            }

            override fun reject(throwable: Throwable, userInfo: WritableMap) {
            }

            override fun reject(code: String, userInfo: WritableMap) {
            }

            override fun reject(code: String, throwable: Throwable?, userInfo: WritableMap) {
            }

            override fun reject(code: String, message: String?, userInfo: WritableMap) {
            }

            override fun reject(
                code: String?,
                message: String?,
                throwable: Throwable?,
                userInfo: WritableMap?
            ) {
            }

            override fun reject(message: String) {
            }

            override fun resolve(value: Any?) {
                Log.d("Llm init", "Llm context initialized successfully!")
                llmStatus = "Llm context loaded successfully!"
                isLlmReady = true
            }
        }

        Log.d("Llm init", "Starting llm context initialization...")
        llmStatus = "Loading model..."
        rnLlm.initContext(1.0, params, promise)
    }

    // A generic function to start a completion, either streaming or not
    private fun startAnyCompletion(isStreaming: Boolean) {
        if (!isLlmReady || isCompleting) return

        isCompleting = true
        completionResult = "" // Reset result at the start
        llmStatus = "Formatting prompt..."

        val messages = listOf(
            mapOf("role" to "system", "content" to "You are a helpful assistant."),
            mapOf("role" to "user", "content" to "Tell me a story about a brave robot.")
        )
        val messagesJson = Gson().toJson(messages)

        val formatParams = Arguments.createMap()

        val formatPromise = object : Promise {
            override fun reject(code: String, message: String?) {
            }

            override fun reject(code: String, throwable: Throwable?) {
            }

            override fun reject(code: String, message: String?, throwable: Throwable?) {
                Log.e("Llm Chat", "Failed to format chat: $message", throwable)
                runOnUiThread {
                    llmStatus = "Error formatting prompt: $message"
                    isCompleting = false
                }
            }

            override fun reject(throwable: Throwable) {
            }

            override fun reject(throwable: Throwable, userInfo: WritableMap) {
            }

            override fun reject(code: String, userInfo: WritableMap) {
            }

            override fun reject(code: String, throwable: Throwable?, userInfo: WritableMap) {
            }

            override fun reject(code: String, message: String?, userInfo: WritableMap) {
            }

            override fun reject(
                code: String?,
                message: String?,
                throwable: Throwable?,
                userInfo: WritableMap?
            ) {
            }

            override fun reject(message: String) {
            }

            override fun resolve(value: Any?) {
                val formattedPrompt = value as String
                runOnUiThread { llmStatus = "Generating response..." }

                if (isStreaming) {
                    runCompletionStream(formattedPrompt)
                } else {
                    runCompletion(formattedPrompt)
                }
            }
        }

        // We use context ID 1.0, the same as in initContext
        rnLlm.getFormattedChat(1.0, messagesJson, null, formatParams, formatPromise)
    }

    private fun runCompletion(prompt: String) {
        val stopWords = Arguments.fromList(listOf("</s>")) // Example stop words

        val completionParams = Arguments.createMap().apply {
            putString("prompt", prompt)
            putInt("n_predict", 100) // Max tokens to generate
            putArray("stop", stopWords)
            putDouble("temperature", 0.7)
            // putBoolean("emit_partial_completion", true)
        }

        val completionPromise = object : Promise {
            override fun resolve(value: Any?) {
                val result = value as WritableMap
                val resultText = result.getString("text") ?: "No text in result"
                val timings = result.getMap("timings")
                val tps = timings?.getDouble("predicted_per_second") ?: 0.0
                tokensPerSecond = tps.roundToInt()

                Log.d("Llm Chat", "Completion finished.")
                Log.d("Llm Chat", "Result text: $resultText")
                if (timings != null) {
                    Log.d(
                        "Llm Chat",
                        "Timings: Predicted tokens: ${timings.getInt("predicted_n")} in ${
                            timings.getInt("predicted_ms")
                        } ms"
                    )
                }

                runOnUiThread {
                    completionResult = resultText.trim()
                    llmStatus = "Completed!"
                    isCompleting = false
                }
            }

            override fun reject(code: String, message: String?) {

            }

            override fun reject(code: String, throwable: Throwable?) {
            }

            override fun reject(code: String, message: String?, throwable: Throwable?) {
                Log.e("Llm Chat", "Completion failed: $message", throwable)
                runOnUiThread {
                    llmStatus = "Error during completion: $message"
                    isCompleting = false
                }
            }

            override fun reject(throwable: Throwable) {
            }

            override fun reject(throwable: Throwable, userInfo: WritableMap) {
            }

            override fun reject(code: String, userInfo: WritableMap) {
            }

            override fun reject(code: String, throwable: Throwable?, userInfo: WritableMap) {
            }

            override fun reject(code: String, message: String?, userInfo: WritableMap) {
            }

            override fun reject(
                code: String?,
                message: String?,
                throwable: Throwable?,
                userInfo: WritableMap?
            ) {
            }

            override fun reject(message: String) {
            }

        }

        rnLlm.completion(1.0, completionParams, completionPromise)
    }

    private fun runCompletionStream(prompt: String) {
        val stopWords = Arguments.fromList(listOf("</s>"))
        val completionParams = Arguments.createMap().apply {
            putString("prompt", prompt)
            putInt("n_predict", 100)
            putArray("stop", stopWords)
            putDouble("temperature", 0.7)
            // putBoolean("emit_partial_completion", true)
        }

        val streamCallback = RNLlama.StreamCallback { token ->
            // Append the new token to our result state
            completionResult += token
            // Log.d("Llm Stream", "Stream finished. Final result map: $completionResult")
        }

        val completionPromise = object : Promise {
            override fun resolve(value: Any?) {
                // This is called when the entire generation is complete
                val result = value as WritableMap
                // Log.d("Llm Stream", "Stream finished. Final result map: $result")
                val timings = result.getMap("timings")
                val tps = timings?.getDouble("predicted_per_second") ?: 0.0
                tokensPerSecond = tps.roundToInt()

                runOnUiThread {
                    llmStatus = "Completed!"
                    isCompleting = false
                }
            }

            override fun reject(code: String, message: String?) {

            }

            override fun reject(code: String, throwable: Throwable?) {
            }

            override fun reject(code: String, message: String?, throwable: Throwable?) {
                Log.e("Llm Stream", "Completion failed: $message", throwable)
                runOnUiThread {
                    llmStatus = "Error during completion: $message"
                    isCompleting = false
                }
            }

            override fun reject(throwable: Throwable) {
            }

            override fun reject(throwable: Throwable, userInfo: WritableMap) {
            }

            override fun reject(code: String, userInfo: WritableMap) {
            }

            override fun reject(code: String, throwable: Throwable?, userInfo: WritableMap) {
            }

            override fun reject(code: String, message: String?, userInfo: WritableMap) {
            }

            override fun reject(
                code: String?,
                message: String?,
                throwable: Throwable?,
                userInfo: WritableMap?
            ) {
            }

            override fun reject(message: String) {
            }
        }

        rnLlm.completionStream(1.0, completionParams, streamCallback, completionPromise)
    }

    // ################################################################
    // Multimodal
    private fun initializeVisionLlmContext() {
        // --- MODIFIED: Add ctx_shift: false for multimodal ---
        val params = Arguments.createMap().apply {
            putString(
                "model",
                "/data/local/tmp/SmolVLM2-500M-Video-Instruct-Q8_0.gguf"
            )
            putInt("n_ctx", 4096)
            // putInt("n_gpu_layers", 99)
            putBoolean("ctx_shift", false) // Crucial for multimodal models
        }

        val promise = object : Promise {
            override fun resolve(value: Any?) {
                Log.d("Llm init", "Llm context initialized successfully!")
                runOnUiThread {
                    llmStatus = "Main context loaded! Initializing multimodal projector..."
                    isLlmReady = true

                    initializeMultimodal()
                }
            }

            override fun reject(code: String, message: String?) {

            }

            override fun reject(code: String, throwable: Throwable?) {
            }

            override fun reject(code: String, message: String?, throwable: Throwable?) {
                Log.e("Llm init", "Failed to initialize Llm context: $message", throwable)
                runOnUiThread { llmStatus = "Error: Failed to load Llm context.\n${message}" }
            }

            override fun reject(throwable: Throwable) {
            }

            override fun reject(throwable: Throwable, userInfo: WritableMap) {
            }

            override fun reject(code: String, userInfo: WritableMap) {
            }

            override fun reject(code: String, throwable: Throwable?, userInfo: WritableMap) {
            }

            override fun reject(code: String, message: String?, userInfo: WritableMap) {
            }

            override fun reject(
                code: String?,
                message: String?,
                throwable: Throwable?,
                userInfo: WritableMap?
            ) {
            }

            override fun reject(message: String) {
            }
        }

        llmStatus = "Loading model..."
        rnLlm.initContext(1.0, params, promise)
    }

    private fun initializeMultimodal() {

        val params = Arguments.createMap().apply {
            putString(
                "path",
                "/data/local/tmp/mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf"
            )
        }

        val promise = object : Promise {
            override fun resolve(value: Any?) {
                if (value as Boolean) {
                    Log.d("Llm init", "Multimodal support initialized successfully!")
                    runOnUiThread {
                        llmStatus = "Model and projector loaded. Ready!"
                        isMultimodalReady = true
                    }
                } else {
                    // reject("initMultimodal returned false.", null)
                }
            }

            override fun reject(code: String, message: String?) {

            }

            override fun reject(code: String, throwable: Throwable?) {
            }

            override fun reject(code: String, message: String?, throwable: Throwable?) {
                Log.e("Llm init", "Failed to init multimodal: $message", throwable)
                runOnUiThread {
                    llmStatus = "Error: Failed to init multimodal projector.\n$message"
                }
            }

            override fun reject(throwable: Throwable) {
            }

            override fun reject(throwable: Throwable, userInfo: WritableMap) {
            }

            override fun reject(code: String, userInfo: WritableMap) {
            }

            override fun reject(code: String, throwable: Throwable?, userInfo: WritableMap) {
            }

            override fun reject(code: String, message: String?, userInfo: WritableMap) {
            }

            override fun reject(
                code: String?,
                message: String?,
                throwable: Throwable?,
                userInfo: WritableMap?
            ) {
            }

            override fun reject(message: String) {
            }
        }

        rnLlm.initMultimodal(1.0, params, promise)
    }

    private fun startVisionCompletion() {
        if (!isMultimodalReady || isCompleting) return

        isCompleting = true
        completionResult = ""
        llmStatus = "Formatting vision prompt..."

//        val messages = listOf(
//            mapOf(
//                "role" to "user",
//                "content" to "What do you see in this image? Describe it in detail.\n<__media__>"
//            )
//
//        )
//        val messages2 = listOf(
//            mapOf("role" to "system", "content" to "You are a helpful assistant."),
//            mapOf("role" to "user", "content" to "What do you see in this image? Describe it in detail.")
//        )

        val messages2 = listOf(
            mapOf(
                "role" to "user",
                "content" to listOf(
                    mapOf(
                        "type" to "text",
                        "text" to "What do you see in this image? Describe it in detail.<__media__>"
                    ),
                    /*mapOf(
                        "type" to "image_url",
                        "image_url" to mapOf(
                            "url" to "file:///data/local/tmp/bike_896.png"
                            // Or: "base64" to "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD..."
                        )
                    )*/
                )
            )
        )
        val messagesJson = Gson().toJson(messages2)

        val formatParams = Arguments.createMap()

        val formatPromise = object : Promise {
            override fun resolve(value: Any?) {
                val formattedPrompt = value as String
                Log.d("Llm Vision", "Formatted Prompt: $formattedPrompt")

                runOnUiThread { llmStatus = "Generating description..." }

                encodeFileToBase64DataUri("/data/local/tmp/lightning.png")?.let {
                    runVisionCompletion(
                        formattedPrompt,
                        it
                    )
                }

                // OR
                /*runVisionCompletion(
                    formattedPrompt,
                    "/data/local/tmp/bike_896.png"
                )*/
            }

            override fun reject(code: String, message: String?) {

            }

            override fun reject(code: String, throwable: Throwable?) {
            }


            override fun reject(code: String, message: String?, throwable: Throwable?) {
                Log.e("Llm Vision", "Failed to format chat: $message", throwable)
                runOnUiThread { isCompleting = false; llmStatus = "Error formatting prompt." }
            }

            override fun reject(throwable: Throwable) {
            }

            override fun reject(throwable: Throwable, userInfo: WritableMap) {
            }

            override fun reject(code: String, userInfo: WritableMap) {
            }

            override fun reject(code: String, throwable: Throwable?, userInfo: WritableMap) {
            }

            override fun reject(code: String, message: String?, userInfo: WritableMap) {
            }

            override fun reject(
                code: String?,
                message: String?,
                throwable: Throwable?,
                userInfo: WritableMap?
            ) {
            }

            override fun reject(message: String) {
            }
        }

        rnLlm.getFormattedChat(1.0, messagesJson, null, formatParams, formatPromise)

        // Just to get information about the multimodal capabilities
        // Use with some modifications above
        // rnLlm.getMultimodalSupport(1.0, formatPromise)
        // Get some info
        // rnLlm.modelInfo()
    }


    private fun runVisionCompletion(prompt: String, mediaFile: String) {
        val stopWords = Arguments.fromList(listOf("</s>", "\n", "User:", "<end_of_utterance>"))

        val paramsImage = Arguments.createMap().apply {
            putString(
                "base64", // or "url
                mediaFile // or file path like file:///path/to/image.jpg
            )
        }

        val paramsAudio = Arguments.createMap().apply {
            putString(
                "format",
                "wav"
            )
        }

        val completionParams = Arguments.createMap().apply {
            putString("prompt", prompt)
            putInt("n_predict", 100)
            // putArray("stop", stopWords)
            putDouble("temperature", 0.1)

            // If an media file is provided, add it to the media_paths array
            val mediaPaths = Arguments.createArray()
            // mediaPaths.pushString("png")
            mediaPaths.pushString(mediaFile)
            // mediaPaths.pushMap(paramsImage)
            putArray("media_paths", mediaPaths)
        }

        val streamCallback = RNLlama.StreamCallback { token ->
            // Append the new token to our result state
            completionResult += token
            // Log.d("Llm Stream", "Stream finished. Final result map: $completionResult")
        }

        val completionPromise = object : Promise {
            override fun resolve(value: Any?) {
                val result = value as WritableMap
                val resultText = result.getString("text") ?: "No text in result"
                val timings = result.getMap("timings")

                val tps = timings?.getDouble("predicted_per_second") ?: 0.0
                tokensPerSecond = tps.roundToInt()

                Log.d("Llm Chat", "Completion finished.")
                Log.d("Llm Chat", "Result text: $resultText")
                if (timings != null) {
                    Log.d(
                        "Llm Chat",
                        "Timings: Predicted tokens: ${timings.getInt("predicted_n")} in ${
                            timings.getInt("predicted_ms")
                        } ms"
                    )
                }

                Log.d("Llm Chat", "Completion finished.")
                runOnUiThread {
                    completionResult = resultText.trim()
                    llmStatus = "Completed!"
                    isCompleting = false
                }
            }

            override fun reject(code: String, message: String?) {

            }

            override fun reject(code: String, throwable: Throwable?) {
            }

            override fun reject(code: String, message: String?, throwable: Throwable?) {
                Log.e("Llm Chat", "Completion failed: $message", throwable)
                runOnUiThread {
                    llmStatus = "Error during completion: $message"
                    isCompleting = false
                }
            }

            override fun reject(throwable: Throwable) {
            }

            override fun reject(throwable: Throwable, userInfo: WritableMap) {
            }

            override fun reject(code: String, userInfo: WritableMap) {
            }

            override fun reject(code: String, throwable: Throwable?, userInfo: WritableMap) {
            }

            override fun reject(code: String, message: String?, userInfo: WritableMap) {
            }

            override fun reject(
                code: String?,
                message: String?,
                throwable: Throwable?,
                userInfo: WritableMap?
            ) {
            }

            override fun reject(message: String) {
            }
        }

        rnLlm.completionStream(1.0, completionParams, streamCallback, completionPromise)
        // OR
        // rnLlm.completion(1.0, completionParams, completionPromise)
    }

    private fun encodeFileToBase64DataUri(filePath: String): String? {
        return try {
            val file = File(filePath)

            if (!file.exists() || !file.canRead()) {
                Log.e("Base64", "File does not exist or cannot be read: $filePath")
                llmStatus = "Error: Could not read file from path."
                return null
            }

            val bytes = file.readBytes()
            val base64String = Base64.encodeToString(bytes, Base64.NO_WRAP)

            val mimeType = when (filePath.substringAfterLast('.').lowercase()) {
                "jpg", "jpeg" -> "image/jpeg"
                "png" -> "image/png"
                "wav" -> "audio/wav"
                "mp3" -> "audio/mpeg"
                // "mp4" -> "video/mp4"
                // "pdf" -> "application/pdf"
                else -> "application/octet-stream" // fallback
            }

            "data:$mimeType;base64,$base64String"
        } catch (e: Exception) {
            Log.e("Base64", "Error encoding file to base64 from path: $filePath", e)
            llmStatus = "Error: Could not encode file."
            null
        }
    }

    // ###########################################
    // Audio
//    private fun initializeAudio() {
////        val whisperModelName = "ggml-tiny.en-q5_1.bin"
////        val whisperFile = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), whisperModelName)
////
////        if (!whisperFile.exists()) {
////            llmStatus = "Error: Whisper model file not found at ${whisperFile.absolutePath}"
////            return
////        }
//
//        val params = Arguments.createMap().apply {
//            putString("path", "/data/local/tmp/mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf")
//        }
//
//        val promise = object : Promise {
//            override fun resolve(value: Any?) {
//                if (value as Boolean) {
//                    Log.d("Llm init", "Audio transcriber (vocoder) initialized successfully!")
//                    runOnUiThread {
//                        llmStatus = "Model, projector, and transcriber loaded. Ready!"
//                        isAudioReady = true
//
//                        initializeMultimodal()
//                    }
//                } else {
//                    reject(null, "initVocoder returned false.", null)
//                }
//            }
//
//            override fun reject(code: String?, message: String?) {
//
//            }
//
//            override fun reject(code: String?, throwable: Throwable?) {
//            }
//
//            override fun reject(code: String?, message: String?, e: Throwable?) {
//                Log.e("Llm init", "Failed to init vocoder: $message", e)
//                runOnUiThread { llmStatus = "Error: Failed to init audio transcriber.\n$message" }
//            }
//
//            override fun reject(throwable: Throwable?) {
//            }
//
//            override fun reject(throwable: Throwable?, userInfo: WritableMap?) {
//            }
//
//            override fun reject(code: String?, userInfo: WritableMap) {
//            }
//
//            override fun reject(code: String?, throwable: Throwable?, userInfo: WritableMap?) {
//            }
//
//            override fun reject(code: String?, message: String?, userInfo: WritableMap) {
//            }
//
//            override fun reject(
//                code: String?,
//                message: String?,
//                throwable: Throwable?,
//                userInfo: WritableMap?
//            ) {
//            }
//
//            override fun reject(message: String?) {
//            }
//        }
//
//        // Note: The library calls this 'initVocoder', but it's used for transcription too.
//        rnLlm.initVocoder(1.0, "mmproj-ultravox-v0_5-llama-3_2-1b-f16.ggu", promise)
//    }
}

@Composable
fun LlmDemoScreen(
    status: String,
    tokensPerSecond: Int,
    result: String,
    isReady: Boolean,
    isMultimodalReady: Boolean,
    isCompleting: Boolean,
    onStartChatCompletion: (Boolean) -> Unit,
    onStartVisionCompletion: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
            .verticalScroll(rememberScrollState()),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Button(
            onClick = { onStartChatCompletion(false) },
            enabled = isReady && !isCompleting
        ) {
            Text(text = "Start Chat Completion")
        }

        Spacer(modifier = Modifier.height(16.dp))

        Button(
            onClick = onStartVisionCompletion,
            enabled = isMultimodalReady && !isCompleting
        ) {
            Text(text = "Start Vision Completion")
        }

        Spacer(modifier = Modifier.height(24.dp))

        Button(
            onClick = { onStartChatCompletion(true) }, // Pass true for streaming
            enabled = isReady && !isCompleting
        ) {
            Text(text = "Start Streaming Chat")
        }

        Spacer(modifier = Modifier.height(24.dp))
        Text(text = status)
        Spacer(modifier = Modifier.height(24.dp))
        Text(text = "T/s = $tokensPerSecond")

        if (isCompleting) {
            CircularProgressIndicator()
        }

        if (result.isNotEmpty()) {
            Text(modifier = Modifier.padding(top = 16.dp), text = "Result:\n$result")
        }
    }
}
