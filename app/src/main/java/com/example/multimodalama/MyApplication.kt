package com.example.multimodalama

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