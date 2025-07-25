package com.example.multimodalama

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
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
        initializeLlamaContext()

        setContent {
            MultimodaLamaTheme {
                LlamaStatusScreen(status = llamaStatus)
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
        val formatParams = Arguments.createMap()

        val formatPromise = object : Promise {
            override fun resolve(value: Any?) {
                val formattedPrompt = value as String
                Log.d("Llama Chat", "Formatted Prompt: $formattedPrompt")
                runOnUiThread {
                    llamaStatus = "Generating response..."
                }
                // --- Step 2: Run Completion ---
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

@Composable
fun LlamaStatusScreen(status: String, modifier: Modifier = Modifier) {
    Box(
        modifier = modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        Text(text = status)
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