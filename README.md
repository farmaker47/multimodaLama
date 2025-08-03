# Android Multimodal LLM Demo with `rn-llama`

This project is a native Android application demonstrating how to run local Large Language Models (LLMs) on-device using the `rn-llama` library. It showcases both standard text-based chat and advanced multimodal (vision) capabilities, all within a modern Android architecture using Kotlin and Jetpack Compose.
It is based on the [llama.rn](https://github.com/mybigday/llama.rn/tree/main) project.

The application serves as a comprehensive example of integrating `llama.cpp` into a native Android project without using a full React Native setup.

## Features

- **Local LLM Inference**: Runs GGUF-formatted models directly on an Android device's CPU.
- **Text & Vision Models**: Demonstrates initialization and inference for both text-only (Gemma) and multimodal vision (SmolVLM2) models.
- **Streaming and Non-Streaming API**:
  - **Standard Completion**: Fetches the entire model response at once.
  - **Streaming Completion**: Receives the response token-by-token for a real-time, chatbot-like experience.
- **Multimodal Prompts**: Shows how to structure prompts that include both text and images for vision-language tasks.
- **Modern Android Stack**: Built with Kotlin, Coroutines, and a declarative UI using Jetpack Compose.
- **Performance Metrics**: Displays the generation speed in tokens per second (T/s).

## How It Works

The project leverages the `rn-llama` library, which provides React Native bindings for the high-performance `llama.cpp` C++ library. Although `rn-llama` is a React Native library, this project demonstrates how to use it in a purely native Android environment.

1.  **Native Library Loading**: The `MyApplication` class initializes `SoLoader`, a utility from Facebook that loads the native `*.so` C++ libraries compiled from `llama.cpp`. This is a crucial first step.
2.  **RNLlama Bridge**: In `MainActivity`, an instance of `RNLlama` is created by providing it with a `ReactApplicationContext`. This acts as the bridge between our Kotlin code and the native C++ functions.
3.  **Model Initialization**:
    -   The `initializeLlmContext()` and `initializeVisionLlmContext()` methods prepare a set of parameters (as a `WritableMap`) specifying the model file path, context size (`n_ctx`), and other configurations.
    -   For vision models, it's critical to set `ctx_shift: false` and load a corresponding multimodal projector (`mmproj`) file.
4.  **Completion Flow**:
    -   A user action triggers a completion (e.g., `startVisionCompletion()`).
    -   The prompt is formatted into the chat structure the model expects using `rnLlm.getFormattedChat()`.
    -   For vision tasks, an image is encoded to a Base64 data URI and passed in the `media_paths` parameter.
    -   The `rnLlm.completionStream()` method is called, which executes the inference on a background thread.
    -   A `StreamCallback` receives generated tokens in real-time, which are appended to a state variable, causing the Jetpack Compose UI to update automatically.

## Setup and Usage

### 1. Obtain Models

This demo is configured to use specific models. Download the following GGUF files and place them on your computer:

-   **Text Model**: [google/gemma-3-1b-it-qat-gguf (Q8_0)](https://huggingface.co/google/gemma-3-1b-it-qat-gguf/blob/main/gemma-3-1b-it-qat-Q8_0.gguf)
-   **Vision Model and Projector**: [SmolVLM-500M-Video-Instruct-GGUF (Q8_0)](https://huggingface.co/ggml-org/SmolVLM2-500M-Video-Instruct-GGUF/tree/main)
-   **Sample Image**: Download any `PNG` or `JPG` image (e.g., from the internet).

### 2. Push Models and Image to Device

The application expects the files to be in the `/data/local/tmp/` directory on your Android device. Use the Android Debug Bridge (`adb`) to push the files.

```sh
# Push the text model
adb push path/to/gemma-3-1b-it-qat-Q8_0.gguf /data/local/tmp/google_gemma-3-1b-it-qat-Q8_0.gguf

# Push the vision model
adb push path/to/SmolVLM2-500M-Video-Instruct-Q8_0.gguf /data/local/tmp/SmolVLM2-500M-Video-Instruct-Q8_0.gguf

# Push the vision projector file
adb push path/to/mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf /data/local/tmp/mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf

# Push a sample image
adb push path/to/your-image.png /data/local/tmp/lightning.png
```

### 3. Build and Run

1.  Open the project in Android Studio.
2.  Build and run the app on a physical Android device (emulators may be slow or lack necessary CPU features).
3.  The app will start, and you can press the buttons to initiate different types of completions.

### 4. Customization

-   To use different models or file paths, modify the hardcoded paths inside `MainActivity.kt`, specifically in `initializeLlmContext()`, `initializeVisionLlmContext()`, and `startVisionCompletion()`.
-   To change the prompts, edit the `messages` lists in `startAnyCompletion()` and `startVisionCompletion()`.

## Key Code Walkthrough

### `MyApplication.kt`
This class ensures the native libraries required by `rn-llama` are loaded when the app starts.

```kotlin
import android.app.Application
import com.facebook.soloader.SoLoader

class MyApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        // Initialize SoLoader. This is the crucial step.
        // It tells the app how to find and load the native libraries.
        SoLoader.init(this, /* native exopackage */ false)
    }
}
```

### `MainActivity.kt` - Initialization
This is how the `RNLlama` library is instantiated and configured for a vision model. Note the use of `ReactApplicationContext` to bridge the native/RN gap.

```kotlin
// In MainActivity class
private lateinit var rnLlm: RNLlama

override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    // The RNLlama class needs a ReactApplicationContext. We can create one from our app context.
    val reactContext = ReactApplicationContext(applicationContext)
    rnLlm = RNLlama(reactContext)

    // ...
    initializeVisionLlmContext()
    //...
}

private fun initializeVisionLlmContext() {
    val params = Arguments.createMap().apply {
        putString(
            "model",
            "/data/local/tmp/SmolVLM2-500M-Video-Instruct-Q8_0.gguf"
        )
        putInt("n_ctx", 4096)
        putBoolean("ctx_shift", false) // Crucial for multimodal models
    }

    // ... rest of the code ...

    rnLlm.initContext(1.0, params, promise)
}

private fun initializeMultimodal() {
    val params = Arguments.createMap().apply {
        putString(
            "path",
            "/data/local/tmp/mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf"
        )
    }
    // ... rest of the code ...
    rnLlm.initMultimodal(1.0, params, promise)
}
```

### `MainActivity.kt` - Streaming Vision Completion
This function demonstrates how to perform a streaming completion with an image.

```kotlin
private fun startVisionCompletion() {
    // ...
    isCompleting = true
    completionResult = ""
    llmStatus = "Formatting vision prompt..."

    // Note the multi-part content with text and an image placeholder
    val messages = listOf(
        mapOf("role" to "user",
              "content" to listOf(
                  mapOf("type" to "text",
                        "text" to "What do you see in this image? Describe it in detail.<__media__>"))))

    val messagesJson = Gson().toJson(messages)

    // ... handle formatted prompt ...

    // After formatting, encode the image and run the completion
    encodeFileToBase64DataUri("/data/local/tmp/lightning.png")?.let {
        runVisionCompletion(formattedPrompt, it)
    }
}

private fun runVisionCompletion(prompt: String, mediaFile: String) {
    val completionParams = Arguments.createMap().apply {
        putString("prompt", prompt)
        putInt("n_predict", 100)
        // ... other params ...
        
        // Pass the Base64-encoded image data
        val mediaPaths = Arguments.createArray()
        mediaPaths.pushString(mediaFile)
        putArray("media_paths", mediaPaths)
    }

    // The callback is invoked for each new token
    val streamCallback = RNLlama.StreamCallback { token ->
        completionResult += token
    }

    // The promise is invoked when the entire stream is finished
    val completionPromise = object : Promise { /* ... */ }

    rnLlm.completionStream(1.0, completionParams, streamCallback, completionPromise)
}
```
