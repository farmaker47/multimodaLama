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
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.core.content.ContextCompat
import com.example.multimodalama.ui.theme.MultimodaLamaTheme
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.WritableMap
import com.rnllama.RNLlama
import java.io.File

class MainActivity : ComponentActivity() {

    private lateinit var rnLlama: RNLlama
    private var llamaStatus = "initializing..."

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
            putString("model", "/data/local/tmp/DeepSeek-R1-ReDistill-Qwen-1.5B-v1.0-Q8_0.gguf")// /data/local/tmp/DeepSeek-R1-ReDistill-Qwen-1.5B-v1.0-Q8_0.gguf
            putInt("n_ctx", 2048) // Context size
            putInt("n_gpu_layers", 0) // GPU layers (0 for CPU on Android for now)
            putBoolean("embedding", false)
        }

        // 5. Create a Promise to handle the async result
        val promise = object : Promise {

            override fun resolve(value: Any?) {
                Log.d("Llama init", "Llama context initialized successfully!")
                llamaStatus = "Llama context loaded successfully!"
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