package com.example.cvproject2

import android.graphics.Color
import android.os.Bundle
import android.widget.ImageButton
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

class GeminiResultsActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_gemini_results)

        window.decorView.setBackgroundColor(Color.parseColor("#1E293B"))

        val btnBack = findViewById<ImageButton>(R.id.btnBack)
        val tvDetectedItems = findViewById<TextView>(R.id.tvDetectedItems)
        val tvGeminiResponse = findViewById<TextView>(R.id.tvGeminiResponse)

        val geminiResponse = intent.getStringExtra("GEMINI_RESPONSE") ?: "No response"
        val detectedObjects = intent.getStringExtra("DETECTED_OBJECTS") ?: "None"

        tvDetectedItems.text = "Detected: $detectedObjects"
        tvGeminiResponse.text = geminiResponse

        btnBack.setOnClickListener {
            finish()
        }
    }
}