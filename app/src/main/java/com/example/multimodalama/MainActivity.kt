package com.example.multimodalama

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Base64
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableDoubleStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
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

    private lateinit var rnLlama: RNLlama
    private var isLlamaReady by mutableStateOf(false)
    private var isCompleting by mutableStateOf(false)
    private var llamaStatus by mutableStateOf("Initializing...")
    private var completionResult by mutableStateOf("")
    private var isMultimodalReady by mutableStateOf(false)
    private var tokensPerSecond by mutableIntStateOf(0)

    // 1. Set up the permission launcher
    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted: Boolean ->
            if (isGranted) {
                Log.d("Llama init", "Storage permission granted")
                initializeLlamaContext()
            } else {
                Log.e("Llama init", "Storage permission denied")
                llamaStatus = "Storage permission denied. Cannot load model."
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // The RNLlama class needs a ReactApplicationContext. We can create one from our app context.
        val reactContext = ReactApplicationContext(applicationContext)
        rnLlama = RNLlama(reactContext)

        // Use either vision OR no vision models.
        // initializeLlamaContext()
        initializeVisionLlamaContext()

        setContent {
            MultimodaLamaTheme {
                LlamaDemoScreen(
                    status = llamaStatus,
                    tokensPerSecond = tokensPerSecond,
                    result = completionResult,
                    isReady = isLlamaReady,
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

    private fun checkAndRequestPermission() {
        when {
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.READ_EXTERNAL_STORAGE
            ) == PackageManager.PERMISSION_GRANTED -> {
                initializeLlamaContext()
            }

            else -> {
                requestPermissionLauncher.launch(Manifest.permission.READ_EXTERNAL_STORAGE)
            }
        }
    }

    private fun initializeLlamaContext() {
        // Multimodals:
        // https://huggingface.co/collections/ggml-org/multimodal-ggufs-68244e01ff1f39e5bebeeedc
        // https://github.com/ggml-org/llama.cpp/tree/master/tools/mtmd#how-to-obtain-mmproj
        // Important: Download the file.gguf AND the mmproj.gguf

        val params: WritableMap = Arguments.createMap().apply {
            putString(
                "model",
                "/data/local/tmp/gemma_greek_2b_it_1000_steps_0_22-q8_0.gguf"
            )// /data/local/tmp/DeepSeek-R1-ReDistill-Qwen-1.5B-v1.0-Q8_0.gguf
            // /data/local/tmp/gemma_greek_2b_it_1000_steps_0_22-q8_0.gguf
            putInt("n_ctx", 2048) // Context size
            putInt("n_gpu_layers", 0) // GPU layers (0 for CPU on Android for now)
            putBoolean("embedding", false)
        }

        val promise = object : Promise {

            override fun resolve(value: Any?) {
                Log.d("Llama init", "Llama context initialized successfully!")
                llamaStatus = "Llama context loaded successfully!"
                isLlamaReady = true
            }

            override fun reject(code: String?, message: String?) {
            }

            override fun reject(code: String?, throwable: Throwable?) {
            }

            override fun reject(code: String?, message: String?, e: Throwable?) {
                Log.e("Llama init", "Failed to initialize Llama context: $code - $message", e)
                runOnUiThread {
                    llamaStatus = "Error: Failed to load Llama context.\n${message}"
                }
            }

            override fun reject(throwable: Throwable?) {
            }

            override fun reject(throwable: Throwable?, userInfo: WritableMap?) {
            }

            override fun reject(code: String?, userInfo: WritableMap) {
            }

            override fun reject(code: String?, throwable: Throwable?, userInfo: WritableMap?) {
            }

            override fun reject(code: String?, message: String?, userInfo: WritableMap) {
            }

            override fun reject(
                code: String?,
                message: String?,
                throwable: Throwable?,
                userInfo: WritableMap?
            ) {
            }

            override fun reject(message: String?) {
            }
        }

        Log.d("Llama init", "Starting llama context initialization...")
        llamaStatus = "Loading model: gemma..."
        rnLlama.initContext(1.0, params, promise)
    }

    // A generic function to start a completion, either streaming or not
    private fun startAnyCompletion(isStreaming: Boolean) {
        if (!isLlamaReady || isCompleting) return

        isCompleting = true
        completionResult = "" // Reset result at the start
        llamaStatus = "Formatting prompt..."

        val messages = listOf(
            mapOf("role" to "system", "content" to "You are a helpful assistant."),
            mapOf("role" to "user", "content" to "Tell me a story about a brave robot.")
        )
        val messagesJson = Gson().toJson(messages)

        val formatParams = Arguments.createMap()

        val formatPromise = object : Promise {
            override fun resolve(value: Any?) {
                val formattedPrompt = value as String
                runOnUiThread { llamaStatus = "Generating response..." }

                if (isStreaming) {
                    runCompletionStream(formattedPrompt)
                } else {
                    runCompletion(formattedPrompt) // The original non-streaming call
                }
            }

            override fun reject(code: String?, message: String?) {

            }

            override fun reject(code: String?, throwable: Throwable?) {
            }

            override fun reject(code: String?, message: String?, e: Throwable?) {
                Log.e("Llama Chat", "Failed to format chat: $message", e)
                runOnUiThread {
                    llamaStatus = "Error formatting prompt: $message"
                    isCompleting = false
                }
            }

            override fun reject(throwable: Throwable?) {
            }

            override fun reject(throwable: Throwable?, userInfo: WritableMap?) {
            }

            override fun reject(code: String?, userInfo: WritableMap) {
            }

            override fun reject(code: String?, throwable: Throwable?, userInfo: WritableMap?) {
            }

            override fun reject(code: String?, message: String?, userInfo: WritableMap) {
            }

            override fun reject(
                code: String?,
                message: String?,
                throwable: Throwable?,
                userInfo: WritableMap?
            ) {
            }

            override fun reject(message: String?) {
            }
        }

        // We use context ID 1.0, the same as in initContext
        rnLlama.getFormattedChat(1.0, messagesJson, null, formatParams, formatPromise)
    }

    private fun runCompletion(prompt: String) {
        val stopWords = Arguments.fromList(listOf("</s>", "\n")) // Example stop words

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

                Log.d("Llama Chat", "Completion finished.")
                Log.d("Llama Chat", "Result text: $resultText")
                if (timings != null) {
                    Log.d(
                        "Llama Chat",
                        "Timings: Predicted tokens: ${timings.getInt("predicted_n")} in ${
                            timings.getInt("predicted_ms")
                        } ms"
                    )
                }

                runOnUiThread {
                    completionResult = resultText.trim()
                    llamaStatus = "Completed!"
                    isCompleting = false
                }
            }

            override fun reject(code: String?, message: String?) {

            }

            override fun reject(code: String?, throwable: Throwable?) {
            }

            override fun reject(code: String?, message: String?, e: Throwable?) {
                Log.e("Llama Chat", "Completion failed: $message", e)
                runOnUiThread {
                    llamaStatus = "Error during completion: $message"
                    isCompleting = false
                }
            }

            override fun reject(throwable: Throwable?) {
            }

            override fun reject(throwable: Throwable?, userInfo: WritableMap?) {
            }

            override fun reject(code: String?, userInfo: WritableMap) {
            }

            override fun reject(code: String?, throwable: Throwable?, userInfo: WritableMap?) {
            }

            override fun reject(code: String?, message: String?, userInfo: WritableMap) {
            }

            override fun reject(
                code: String?,
                message: String?,
                throwable: Throwable?,
                userInfo: WritableMap?
            ) {
            }

            override fun reject(message: String?) {
            }
        }

        rnLlama.completion(1.0, completionParams, completionPromise)
    }

    private fun runCompletionStream(prompt: String) {
        val stopWords = Arguments.fromList(listOf("</s>", "\n"))
        val completionParams = Arguments.createMap().apply {
            putString("prompt", prompt)
            putInt("n_predict", 100)
            putArray("stop", stopWords)
            putDouble("temperature", 0.7)
            // putBoolean("emit_partial_completion", true)
        }

        val streamCallback = object : RNLlama.StreamCallback {
            override fun onToken(token: String) {
                // Append the new token to our result state
                completionResult += token
                // Log.d("Llama Stream", "Stream finished. Final result map: $completionResult")
            }
        }

        val completionPromise = object : Promise {
            override fun resolve(value: Any?) {
                // This is called when the entire generation is complete
                val result = value as WritableMap
                // Log.d("Llama Stream", "Stream finished. Final result map: $result")
                val timings = result.getMap("timings")
                val tps = timings?.getDouble("predicted_per_second") ?: 0.0
                tokensPerSecond = tps.roundToInt()

                runOnUiThread {
                    llamaStatus = "Completed!"
                    isCompleting = false
                }
            }

            override fun reject(code: String?, message: String?) {

            }

            override fun reject(code: String?, throwable: Throwable?) {
            }

            override fun reject(code: String?, message: String?, e: Throwable?) {
                Log.e("Llama Stream", "Completion failed: $message", e)
                runOnUiThread {
                    llamaStatus = "Error during completion: $message"
                    isCompleting = false
                }
            }

            override fun reject(throwable: Throwable?) {
            }

            override fun reject(throwable: Throwable?, userInfo: WritableMap?) {
            }

            override fun reject(code: String?, userInfo: WritableMap) {
            }

            override fun reject(code: String?, throwable: Throwable?, userInfo: WritableMap?) {
            }

            override fun reject(code: String?, message: String?, userInfo: WritableMap) {
            }

            override fun reject(
                code: String?,
                message: String?,
                throwable: Throwable?,
                userInfo: WritableMap?
            ) {
            }

            override fun reject(message: String?) {
            }
        }

        // 3. Call the new streaming method
        rnLlama.completionStream(1.0, completionParams, streamCallback, completionPromise)
    }

    // ################################################################
    // Multimodal
    private fun initializeVisionLlamaContext() {
        // --- MODIFIED: Add ctx_shift: false for multimodal ---
        val params = Arguments.createMap().apply {
            putString("model", "/data/local/tmp/SmolVLM2-500M-Video-Instruct-Q8_0.gguf")
            putInt("n_ctx", 4096)
            // putInt("n_gpu_layers", 99)
            putBoolean("ctx_shift", false) // Crucial for multimodal models
        }

        val promise = object : Promise {
            override fun resolve(value: Any?) {
                Log.d("Llama init", "Llama context initialized successfully!")
                runOnUiThread {
                    llamaStatus = "Main context loaded! Initializing multimodal projector..."
                    isLlamaReady = true

                    initializeMultimodal()
                }
            }

            override fun reject(code: String?, message: String?) {

            }

            override fun reject(code: String?, throwable: Throwable?) {
            }

            override fun reject(code: String?, message: String?, e: Throwable?) {
                Log.e("Llama init", "Failed to initialize Llama context: $message", e)
                runOnUiThread { llamaStatus = "Error: Failed to load Llama context.\n${message}" }
            }

            override fun reject(throwable: Throwable?) {
            }

            override fun reject(throwable: Throwable?, userInfo: WritableMap?) {
            }

            override fun reject(code: String?, userInfo: WritableMap) {
            }

            override fun reject(code: String?, throwable: Throwable?, userInfo: WritableMap?) {
            }

            override fun reject(code: String?, message: String?, userInfo: WritableMap) {
            }

            override fun reject(
                code: String?,
                message: String?,
                throwable: Throwable?,
                userInfo: WritableMap?
            ) {
            }

            override fun reject(message: String?) {
            }
        }

        llamaStatus = "Loading model: SmolVLM2..."
        rnLlama.initContext(1.0, params, promise)
    }

    private fun initializeMultimodal() {

        val params = Arguments.createMap().apply {
            putString("path", "/data/local/tmp/mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf")
        }

        val promise = object : Promise {
            override fun resolve(value: Any?) {
                if (value as Boolean) {
                    Log.d("Llama init", "Multimodal support initialized successfully!")
                    runOnUiThread {
                        llamaStatus = "Model and projector loaded. Ready!"
                        isMultimodalReady = true
                    }
                } else {
                    reject(null, "initMultimodal returned false.", null)
                }
            }

            override fun reject(code: String?, message: String?) {

            }

            override fun reject(code: String?, throwable: Throwable?) {
            }

            override fun reject(code: String?, message: String?, e: Throwable?) {
                Log.e("Llama init", "Failed to init multimodal: $message", e)
                runOnUiThread {
                    llamaStatus = "Error: Failed to init multimodal projector.\n$message"
                }
            }

            override fun reject(throwable: Throwable?) {
            }

            override fun reject(throwable: Throwable?, userInfo: WritableMap?) {
            }

            override fun reject(code: String?, userInfo: WritableMap) {
            }

            override fun reject(code: String?, throwable: Throwable?, userInfo: WritableMap?) {
            }

            override fun reject(code: String?, message: String?, userInfo: WritableMap) {
            }

            override fun reject(
                code: String?,
                message: String?,
                throwable: Throwable?,
                userInfo: WritableMap?
            ) {
            }

            override fun reject(message: String?) {
            }
        }

        rnLlama.initMultimodal(1.0, params, promise)
    }

    private fun startVisionCompletion() {
        if (!isMultimodalReady || isCompleting) return

        isCompleting = true
        completionResult = ""
        llamaStatus = "Formatting vision prompt..."

        val messages = listOf(
            mapOf(
                "role" to "user",
                "content" to "What do you see in this image? Describe it in detail.\n<__media__>"
            )
        )
        val messagesJson = Gson().toJson(messages)

        val formatParams = Arguments.createMap()

        val formatPromise = object : Promise {
            override fun resolve(value: Any?) {
                val formattedPrompt = value as String
                Log.d("Llama Vision", "Formatted Prompt: $formattedPrompt")

                runOnUiThread { llamaStatus = "Generating description..." }

                encodeFileToBase64DataUri("/data/local/tmp/bike_896.png")?.let {
                    runVisionCompletion(
                        formattedPrompt,
                        it//    OR just     "/data/local/tmp/bike_896.png"
                    )
                }
            }

            override fun reject(code: String?, message: String?) {

            }

            override fun reject(code: String?, throwable: Throwable?) {
            }

            override fun reject(code: String?, message: String?, e: Throwable?) {
                Log.e("Llama Vision", "Failed to format chat: $message", e)
                runOnUiThread { isCompleting = false; llamaStatus = "Error formatting prompt." }
            }

            override fun reject(throwable: Throwable?) {
            }

            override fun reject(throwable: Throwable?, userInfo: WritableMap?) {
            }

            override fun reject(code: String?, userInfo: WritableMap) {
            }

            override fun reject(code: String?, throwable: Throwable?, userInfo: WritableMap?) {
            }

            override fun reject(code: String?, message: String?, userInfo: WritableMap) {
            }

            override fun reject(
                code: String?,
                message: String?,
                throwable: Throwable?,
                userInfo: WritableMap?
            ) {
            }

            override fun reject(message: String?) {
            }
        }

        rnLlama.getFormattedChat(1.0, messagesJson, null, formatParams, formatPromise)
    }


    private fun runVisionCompletion(prompt: String, imageFile: String) {
        val stopWords = Arguments.fromList(listOf("</s>", "\n", "User:", "<end_of_utterance>"))

        val completionParams = Arguments.createMap().apply {
            putString("prompt", prompt)
            putInt("n_predict", 100)
            putArray("stop", stopWords)
            putDouble("temperature", 0.1)

            // If an image file is provided, add it to the media_paths array
            val mediaPaths = Arguments.createArray()
            mediaPaths.pushString(imageFile)
            putArray("media_paths", mediaPaths)
        }

        val streamCallback = object : RNLlama.StreamCallback {
            override fun onToken(token: String) {
                // Append the new token to our result state
                completionResult += token
                // Log.d("Llama Stream", "Stream finished. Final result map: $completionResult")
            }
        }

        val completionPromise = object : Promise {
            override fun resolve(value: Any?) {
                val result = value as WritableMap
                val resultText = result.getString("text") ?: "No text in result"
                val timings = result.getMap("timings")

                val tps = timings?.getDouble("predicted_per_second") ?: 0.0
                tokensPerSecond = tps.roundToInt()

                Log.d("Llama Chat", "Completion finished.")
                Log.d("Llama Chat", "Result text: $resultText")
                if (timings != null) {
                    Log.d(
                        "Llama Chat",
                        "Timings: Predicted tokens: ${timings.getInt("predicted_n")} in ${
                            timings.getInt("predicted_ms")
                        } ms"
                    )
                }

                Log.d("Llama Chat", "Completion finished.")
                runOnUiThread {
                    completionResult = resultText.trim()
                    llamaStatus = "Completed!"
                    isCompleting = false
                }
            }

            override fun reject(code: String?, message: String?) {
            }

            override fun reject(code: String?, throwable: Throwable?) {
            }

            override fun reject(code: String?, message: String?, e: Throwable?) {
                Log.e("Llama Chat", "Completion failed: $message", e)
                runOnUiThread {
                    llamaStatus = "Error during completion: $message"
                    isCompleting = false
                }
            }

            override fun reject(throwable: Throwable?) {
            }

            override fun reject(throwable: Throwable?, userInfo: WritableMap?) {
            }

            override fun reject(code: String?, userInfo: WritableMap) {
            }

            override fun reject(code: String?, throwable: Throwable?, userInfo: WritableMap?) {
            }

            override fun reject(code: String?, message: String?, userInfo: WritableMap) {
            }

            override fun reject(
                code: String?,
                message: String?,
                throwable: Throwable?,
                userInfo: WritableMap?
            ) {
            }

            override fun reject(message: String?) {
            }
        }

        rnLlama.completionStream(1.0, completionParams, streamCallback, completionPromise)
    }

    private fun encodeFileToBase64DataUri(filePath: String): String? {
        return try {
            val file = File(filePath)

            // Add a check to ensure the file exists and is readable
            if (!file.exists() || !file.canRead()) {
                Log.e("Base64", "File does not exist or cannot be read: $filePath")
                llamaStatus = "Error: Could not read image from path."
                return null
            }

            // Use the file path to read bytes directly
            val bytes = file.readBytes()
            val base64String = Base64.encodeToString(bytes, Base64.NO_WRAP)

            // This logic works the same, using the file path to get the extension
            val mimeType = when (filePath.substringAfterLast('.').lowercase()) {
                "jpg", "jpeg" -> "image/jpeg"
                "png" -> "image/png"
                else -> "application/octet-stream" // fallback
            }

            "data:$mimeType;base64,$base64String"
        } catch (e: Exception) { // Catch broader exceptions like SecurityException
            Log.e("Base64", "Error encoding file to base64 from path: $filePath", e)
            llamaStatus = "Error: Could not encode image."
            null
        }
    }
}

@Composable
fun LlamaDemoScreen(
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
            .padding(16.dp),
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
