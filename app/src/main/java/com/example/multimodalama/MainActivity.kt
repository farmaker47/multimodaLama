package com.example.multimodalama

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
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
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.example.multimodalama.ui.theme.MultimodaLamaTheme
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.WritableMap
import com.google.gson.Gson
import com.rnllama.RNLlama
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {

    private lateinit var rnLlama: RNLlama
    private var isLlamaReady by mutableStateOf(false)
    private var isCompleting by mutableStateOf(false)
    private var llamaStatus by mutableStateOf("Initializing...")
    private var completionResult by mutableStateOf("")
    private var isMultimodalReady by mutableStateOf(false) // New state for multimodal

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

        // 2. Check for and request permission
        // checkAndRequestPermission()
        // initializeLlamaContext()
        initializeVisionLlamaContext()

        setContent {
            MultimodaLamaTheme {
                LlamaDemoScreen(
                    status = llamaStatus,
                    result = completionResult,
                    isReady = isLlamaReady,
                    isMultimodalReady = isMultimodalReady, // Pass new state to UI
                    isCompleting = isCompleting,
                    onStartChatCompletion = { startChatCompletion() },
                    onStartVisionCompletion = { startVisionCompletion() } // New event handler
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
        // 3. Define the path to your model file
//        val modelName = "phi-2.Q4_K_M.gguf" // Change this to your model's name
//        val modelFile = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), modelName)
//
//        if (!modelFile.exists()) {
//            Log.e("Llama init", "Model file does not exist at: ${modelFile.absolutePath}")
//            llamaStatus = "Error: Model file not found!"
//            return
//        }

        // 4. Create the parameters map (like a Bundle for React Native)

        // Multimodals:
        // https://huggingface.co/collections/ggml-org/multimodal-ggufs-68244e01ff1f39e5bebeeedc
        // https://github.com/ggml-org/llama.cpp/tree/master/tools/mtmd#how-to-obtain-mmproj
        // Important: Download the file.gguf AND the mmproj.gguf

        val params: WritableMap = Arguments.createMap().apply {
            putString("model", "/data/local/tmp/gemma_greek_2b_it_1000_steps_0_22-q8_0.gguf")// /data/local/tmp/DeepSeek-R1-ReDistill-Qwen-1.5B-v1.0-Q8_0.gguf
            // /data/local/tmp/gemma_greek_2b_it_1000_steps_0_22-q8_0.gguf
            putInt("n_ctx", 2048) // Context size
            putInt("n_gpu_layers", 0) // GPU layers (0 for CPU on Android for now)
            putBoolean("embedding", false)
        }

        // 5. Create a Promise to handle the async result
        val promise = object : Promise {

            override fun resolve(value: Any?) {
                Log.d("Llama init", "Llama context initialized successfully!")
                llamaStatus = "Llama context loaded successfully!"
                isLlamaReady = true

                lifecycleScope.launch {
                    // delay(1000)
                    startChatCompletion()
                }
            }

            override fun reject(code: String?, message: String?) {
                TODO("Not yet implemented")
            }

            override fun reject(code: String?, throwable: Throwable?) {
                TODO("Not yet implemented")
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

        // 6. Call initContext. The RNLlama class runs this on a background thread.
        // We use a unique ID (e.g., 1.0) for this context.
        Log.d("Llama init", "Starting llama context initialization...")
        llamaStatus = "Loading model: deepseek..."
        rnLlama.initContext(1.0, params, promise)
    }

    private fun startChatCompletion() {
        if (!isLlamaReady || isCompleting) return

        isCompleting = true
        completionResult = ""
        llamaStatus = "Formatting prompt..."

        // --- Step 1: Format the Chat Prompt ---

        // Create the message structure as a list of maps
        val messages = listOf(
            mapOf("role" to "system", "content" to "You are a helpful assistant."),
            mapOf("role" to "user", "content" to "Hello! Can you tell me a short story?")
        )

        // The library expects this as a JSON string
        val messagesJson = Gson().toJson(messages)

        // Create an empty params map for getFormattedChat
        val formatParams = Arguments.createMap().apply {
            // Add these two lines to use the advanced formatter and disable thinking
            putBoolean("jinja", true)
            putBoolean("enable_thinking", false)
        }

        val formatPromise = object : Promise {
            override fun resolve(value: Any?) {
                // 1. Cast the result to WritableMap, not String
                val formatResult = value as WritableMap

                // 2. Extract the "prompt" string from the map
                val formattedPrompt = formatResult.getString("prompt") ?: ""

                if (formattedPrompt.isEmpty()) {
                    Log.e("Llama Chat", "Formatted prompt was empty!")
                    runOnUiThread {
                        llamaStatus = "Error: Failed to generate a valid prompt."
                        isCompleting = false
                    }
                    return
                }

                Log.d("Llama Chat", "Formatted Prompt: $formattedPrompt")
                runOnUiThread {
                    llamaStatus = "Generating response..."
                }
                // Now run completion with the extracted prompt
                runCompletion(formattedPrompt)
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
        }

        val completionPromise = object : Promise {
            override fun resolve(value: Any?) {
                val result = value as WritableMap
                val resultText = result.getString("text") ?: "No text in result"
                val timings = result.getMap("timings")

                Log.d("Llama Chat", "Completion finished.")
                Log.d("Llama Chat", "Result text: $resultText")
                if (timings != null) {
                    Log.d("Llama Chat", "Timings: Predicted tokens: ${timings.getInt("predicted_n")} in ${timings.getInt("predicted_ms")} ms")
                }

                runOnUiThread {
                    completionResult = resultText.trim()
                    llamaStatus = "Completed!\n $resultText"
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



    // ################################################################
    private fun initializeVisionLlamaContext() {
        // --- MODIFIED: Use multimodal model files ---
//        val modelName = "llava-v1.5-7b-q4_k_m.gguf"
//        val modelFile = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), modelName)
//
//        if (!modelFile.exists()) {
//            llamaStatus = "Error: Model file not found at ${modelFile.absolutePath}"
//            return
//        }

        // --- MODIFIED: Add ctx_shift: false for multimodal ---
        val params = Arguments.createMap().apply {
            putString("model", "/data/local/tmp/SmolVLM2-500M-Video-Instruct-Q8_0.gguf")
            putInt("n_ctx", 4096)
            putBoolean("ctx_shift", false) // Crucial for multimodal models
        }

        val promise = object : Promise {
            override fun resolve(value: Any?) {
                Log.d("Llama init", "Llama context initialized successfully!")
                runOnUiThread {
                    llamaStatus = "Main context loaded! Initializing multimodal projector..."
                    isLlamaReady = true
                    // --- NEW: Chain the multimodal initialization ---
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

    // --- NEW: Function to initialize the multimodal projector ---
    private fun initializeMultimodal() {
//        val mmprojName = "mmproj-model-f16.gguf"
//        val mmprojFile = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), mmprojName)
//
//        if (!mmprojFile.exists()) {
//            llamaStatus = "Error: MMPROJ file not found at ${mmprojFile.absolutePath}"
//            return
//        }

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
                runOnUiThread { llamaStatus = "Error: Failed to init multimodal projector.\n$message" }
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

        // --- FIX IS HERE: Part 1 ---

        // The messages JSON now uses the <__media__> placeholder.
        // The actual image path is passed later in runCompletion.
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

                // We now pass the image file path to the completion function
                //val imageName = "cat.jpg"
                //val imageFile = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), imageName)

                runOnUiThread { llamaStatus = "Generating description..." }
                runVisionCompletion(formattedPrompt, "/data/local/tmp/bike_896.png") // Pass the image file
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
        val stopWords = Arguments.fromList(listOf("</s>", "\n", "User:"))

        val completionParams = Arguments.createMap().apply {
            putString("prompt", prompt)
            putInt("n_predict", 200)
            putArray("stop", stopWords)
            putDouble("temperature", 0.1)

            // If an image file is provided, add it to the media_paths array
            val mediaPaths = Arguments.createArray()
            mediaPaths.pushString(imageFile)
            putArray("media_paths", mediaPaths)
        }

        val completionPromise = object : Promise {
            override fun resolve(value: Any?) {
                val result = value as WritableMap
                val resultText = result.getString("text") ?: "No text in result"
                val timings = result.getMap("timings")

                Log.d("Llama Chat", "Completion finished.")
                Log.d("Llama Chat", "Result text: $resultText")
                if (timings != null) {
                    Log.d("Llama Chat", "Timings: Predicted tokens: ${timings.getInt("predicted_n")} in ${timings.getInt("predicted_ms")} ms")
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

        rnLlama.completion(1.0, completionParams, completionPromise)
    }
}

// --- MODIFIED: Update the UI Composable ---
@Composable
fun LlamaDemoScreen(
    status: String,
    result: String,
    isReady: Boolean,
    isMultimodalReady: Boolean,
    isCompleting: Boolean,
    onStartChatCompletion: () -> Unit,
    onStartVisionCompletion: () -> Unit
) {
    Column(
        modifier = Modifier.fillMaxSize().padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(text = status)
        Spacer(modifier = Modifier.height(24.dp))

        Button(
            onClick = onStartChatCompletion,
            enabled = isReady && !isCompleting
        ) {
            Text(text = "Start Chat Completion")
        }

        Spacer(modifier = Modifier.height(16.dp))

        Button(
            onClick = onStartVisionCompletion,
            enabled = isMultimodalReady && !isCompleting // Enabled only when projector is also loaded
        ) {
            Text(text = "Start Vision Completion")
        }

        Spacer(modifier = Modifier.height(24.dp))

        if (isCompleting) {
            CircularProgressIndicator()
        }

        if (result.isNotEmpty()) {
            Text(modifier = Modifier.padding(top = 16.dp), text = "Result:\n$result")
        }
    }
}

@Composable
fun Greeting(name: String, modifier: Modifier = Modifier) {
    Text(
        text = "Hello $name!",
        modifier = modifier
    )
}

@Preview(showBackground = true)
@Composable
fun GreetingPreview() {
    MultimodaLamaTheme {
        Greeting("Android")
    }
}