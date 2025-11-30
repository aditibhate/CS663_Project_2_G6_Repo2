package com.example.cvproject2

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.*
import android.net.Uri
import android.os.Bundle
import android.util.Base64
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.exifinterface.media.ExifInterface
import ai.onnxruntime.*
import kotlinx.coroutines.*
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import kotlin.math.max
import kotlin.math.min

class MainActivity : AppCompatActivity() {

    private lateinit var imageView: ImageView
    private lateinit var btnCamera: Button
    private lateinit var btnGallery: Button
    private lateinit var btnDetect: Button
    private lateinit var btnGemini: Button
    private lateinit var tvResults: TextView
    private var currentPhotoUri: Uri? = null
    private var currentBitmap: Bitmap? = null
    private var detectedObjects = listOf<Detection>()

    private lateinit var ortEnvironment: OrtEnvironment
    private lateinit var ortSession: OrtSession

    // YOLOv8 configuration
    private val inputSize = 640
    private val confidenceThreshold = 0.25f
    private val iouThreshold = 0.45f

    // Add your Gemini API key here
    private val GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"

    // Waste classification classes
    private val classNames = listOf(
        "biowaste", "glass", "metal", "other", "paper", "plastic", "styrofoam", "wrapper"
    )

    private val cameraLauncher = registerForActivityResult(
        ActivityResultContracts.TakePicture()
    ) { success ->
        if (success) {
            loadImage(currentPhotoUri)
        }
    }

    private val galleryLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        uri?.let {
            loadImage(it)
        }
    }

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            launchCamera()
        } else {
            Toast.makeText(this, "Camera permission required", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        window.decorView.setBackgroundColor(Color.parseColor("#1E293B"))

        imageView = findViewById(R.id.imageView)
        btnCamera = findViewById(R.id.btnCamera)
        btnGallery = findViewById(R.id.btnGallery)
        btnDetect = findViewById(R.id.btnDetect)
        btnGemini = findViewById(R.id.btnGemini)
        tvResults = findViewById(R.id.tvResults)

        btnGemini.isEnabled = false

        initOnnxModel()

        btnCamera.setOnClickListener {
            if (checkCameraPermission()) {
                launchCamera()
            } else {
                requestCameraPermission()
            }
        }

        btnGallery.setOnClickListener {
            galleryLauncher.launch("image/*")
        }

        btnDetect.setOnClickListener {
            currentBitmap?.let {
                runObjectDetection(it)
            } ?: Toast.makeText(this, "Please capture an image first", Toast.LENGTH_SHORT).show()
        }

        btnGemini.setOnClickListener {
            if (detectedObjects.isNotEmpty() && currentBitmap != null) {
                getGeminiRecyclingAdvice(currentBitmap!!, detectedObjects)
            } else {
                Toast.makeText(this, "Please detect objects first", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun initOnnxModel() {
        try {
            ortEnvironment = OrtEnvironment.getEnvironment()
            val modelBytes = assets.open("trash.onnx").readBytes()
            ortSession = ortEnvironment.createSession(modelBytes)
            Toast.makeText(this, "Model loaded successfully", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Toast.makeText(this, "Error loading model: ${e.message}", Toast.LENGTH_LONG).show()
            e.printStackTrace()
        }
    }

    private fun loadImage(uri: Uri?) {
        try {
            uri?.let {
                val inputStream = contentResolver.openInputStream(it)
                var bitmap = BitmapFactory.decodeStream(inputStream)
                inputStream?.close()

                bitmap = fixImageRotation(it, bitmap)

                currentBitmap = bitmap
                imageView.setImageBitmap(bitmap)
                tvResults.text = "Image loaded. Tap 'Detect Objects' to analyze."
                tvResults.setTextColor(Color.parseColor("#94A3B8"))
            }
        } catch (e: Exception) {
            Toast.makeText(this, "Error loading image: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun fixImageRotation(uri: Uri, bitmap: Bitmap): Bitmap {
        val inputStream = contentResolver.openInputStream(uri)
        val exif = inputStream?.let { ExifInterface(it) }
        inputStream?.close()

        val orientation = exif?.getAttributeInt(
            ExifInterface.TAG_ORIENTATION,
            ExifInterface.ORIENTATION_NORMAL
        ) ?: ExifInterface.ORIENTATION_NORMAL

        val matrix = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
        }

        return if (orientation != ExifInterface.ORIENTATION_NORMAL) {
            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        } else {
            bitmap
        }
    }

    private fun runObjectDetection(bitmap: Bitmap) {
        try {
            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
            val inputTensor = preprocessImage(resizedBitmap)

            val inputName = ortSession.inputNames.iterator().next()
            val output = ortSession.run(mapOf(inputName to inputTensor))

            val outputTensor = output[0].value as Array<*>

            val detections = processYoloV8Output(outputTensor, bitmap.width, bitmap.height)
            detectedObjects = detections

            val resultBitmap = drawBoundingBoxes(bitmap, detections)
            imageView.setImageBitmap(resultBitmap)

            displayDetectionResults(detections)

            btnGemini.isEnabled = detections.isNotEmpty()

            Toast.makeText(this, "Detected ${detections.size} objects", Toast.LENGTH_SHORT).show()

        } catch (e: Exception) {
            Toast.makeText(this, "Detection error: ${e.message}", Toast.LENGTH_LONG).show()
            e.printStackTrace()
        }
    }

    private fun preprocessImage(bitmap: Bitmap): OnnxTensor {
        val floatBuffer = java.nio.FloatBuffer.allocate(3 * inputSize * inputSize)
        val pixels = IntArray(inputSize * inputSize)
        bitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        for (c in 0..2) {
            for (y in 0 until inputSize) {
                for (x in 0 until inputSize) {
                    val pixel = pixels[y * inputSize + x]
                    val value = when (c) {
                        0 -> Color.red(pixel) / 255f
                        1 -> Color.green(pixel) / 255f
                        else -> Color.blue(pixel) / 255f
                    }
                    floatBuffer.put(value)
                }
            }
        }

        floatBuffer.rewind()
        val shape = longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
        return OnnxTensor.createTensor(ortEnvironment, floatBuffer, shape)
    }

    private fun processYoloV8Output(
        output: Array<*>,
        originalWidth: Int,
        originalHeight: Int
    ): List<Detection> {
        val predictions = output[0] as Array<*>
        val numDetections = (predictions[0] as FloatArray).size
        val numClasses = classNames.size

        val detections = mutableListOf<Detection>()

        for (i in 0 until numDetections) {
            val cx = (predictions[0] as FloatArray)[i]
            val cy = (predictions[1] as FloatArray)[i]
            val w = (predictions[2] as FloatArray)[i]
            val h = (predictions[3] as FloatArray)[i]

            var maxScore = 0f
            var maxIndex = 0
            for (j in 0 until numClasses) {
                val score = (predictions[4 + j] as FloatArray)[i]
                if (score > maxScore) {
                    maxScore = score
                    maxIndex = j
                }
            }

            if (maxScore > confidenceThreshold) {
                val x1 = (cx - w / 2) / inputSize * originalWidth
                val y1 = (cy - h / 2) / inputSize * originalHeight
                val x2 = (cx + w / 2) / inputSize * originalWidth
                val y2 = (cy + h / 2) / inputSize * originalHeight

                detections.add(
                    Detection(
                        x1 = x1.coerceIn(0f, originalWidth.toFloat()),
                        y1 = y1.coerceIn(0f, originalHeight.toFloat()),
                        x2 = x2.coerceIn(0f, originalWidth.toFloat()),
                        y2 = y2.coerceIn(0f, originalHeight.toFloat()),
                        confidence = maxScore,
                        classIndex = maxIndex,
                        className = if (maxIndex < classNames.size) classNames[maxIndex] else "Unknown"
                    )
                )
            }
        }

        return applyNMS(detections)
    }

    private fun applyNMS(detections: List<Detection>): List<Detection> {
        val sortedDetections = detections.sortedByDescending { it.confidence }.toMutableList()
        val selectedDetections = mutableListOf<Detection>()

        while (sortedDetections.isNotEmpty()) {
            val best = sortedDetections.removeAt(0)
            selectedDetections.add(best)

            sortedDetections.removeAll { detection ->
                calculateIoU(best, detection) > iouThreshold &&
                        best.classIndex == detection.classIndex
            }
        }

        return selectedDetections
    }

    private fun calculateIoU(box1: Detection, box2: Detection): Float {
        val x1 = max(box1.x1, box2.x1)
        val y1 = max(box1.y1, box2.y1)
        val x2 = min(box1.x2, box2.x2)
        val y2 = min(box1.y2, box2.y2)

        val intersectionArea = max(0f, x2 - x1) * max(0f, y2 - y1)
        val box1Area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
        val box2Area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
        val unionArea = box1Area + box2Area - intersectionArea

        return if (unionArea > 0) intersectionArea / unionArea else 0f
    }

    private fun drawBoundingBoxes(bitmap: Bitmap, detections: List<Detection>): Bitmap {
        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)

        val paint = Paint().apply {
            style = Paint.Style.STROKE
            strokeWidth = 8f
            textSize = 50f
            typeface = Typeface.DEFAULT_BOLD
        }

        val textPaint = Paint().apply {
            color = Color.WHITE
            textSize = 50f
            typeface = Typeface.DEFAULT_BOLD
        }

        val colors = listOf(
            Color.rgb(255, 0, 0),
            Color.rgb(0, 255, 0),
            Color.rgb(0, 0, 255),
            Color.rgb(255, 255, 0),
            Color.rgb(255, 0, 255),
            Color.rgb(0, 255, 255),
            Color.rgb(255, 128, 0),
            Color.rgb(128, 0, 255),
            Color.rgb(255, 192, 203),
            Color.rgb(0, 255, 128)
        )

        detections.forEachIndexed { index, detection ->
            val color = colors[index % colors.size]
            paint.color = color

            canvas.drawRect(
                detection.x1,
                detection.y1,
                detection.x2,
                detection.y2,
                paint
            )

            val label = "${detection.className} ${(detection.confidence * 100).toInt()}%"
            val textBounds = Rect()
            textPaint.getTextBounds(label, 0, label.length, textBounds)

            val backgroundPaint = Paint().apply {
                this.color = color
                style = Paint.Style.FILL
            }

            canvas.drawRect(
                detection.x1,
                detection.y1 - textBounds.height() - 20,
                detection.x1 + textBounds.width() + 30,
                detection.y1,
                backgroundPaint
            )

            canvas.drawText(
                label,
                detection.x1 + 15,
                detection.y1 - 15,
                textPaint
            )
        }

        return mutableBitmap
    }

    private fun displayDetectionResults(detections: List<Detection>) {
        if (detections.isEmpty()) {
            tvResults.text = "❌ No objects detected"
            tvResults.setTextColor(Color.parseColor("#EF4444"))
            return
        }

        val objectCounts = detections.groupBy { it.className }
            .map { (className, list) -> "$className: ${list.size}" }
            .sorted()

        val colors = listOf(
            "🔴", "🟢", "🔵", "🟡", "🟣", "🔵", "🟠", "🟣", "🌸", "🟢"
        )

        val resultText = buildString {
            append("✅ Detected Objects:\n\n")
            detections.forEachIndexed { index, detection ->
                val emoji = colors[index % colors.size]
                append("$emoji ${detection.className} - ${(detection.confidence * 100).toInt()}%\n")
            }
            append("\n📊 Summary:\n")
            objectCounts.forEach { count ->
                append("• $count\n")
            }
        }

        tvResults.text = resultText
        tvResults.setTextColor(Color.parseColor("#10B981"))
    }

    private fun getGeminiRecyclingAdvice(bitmap: Bitmap, detections: List<Detection>) {
        if (GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE") {
            Toast.makeText(this, "Please add your Gemini API key", Toast.LENGTH_LONG).show()
            return
        }

        btnGemini.isEnabled = false
        btnGemini.text = "Loading..."
        tvResults.append("\n\n🤖 Asking Gemini AI...\n")

        CoroutineScope(Dispatchers.IO).launch {
            try {
                val base64Image = bitmapToBase64(bitmap)
                val objectsList = detections.joinToString(", ") { it.className }

                // THIS IS THE PROMPT - YOU CAN MODIFY THIS
                val prompt = """
                    I have detected the following waste items in this image: $objectsList
                    
                    For each item detected, please provide:
                    1. Is it recyclable? (Yes/No)
                    2. How to recycle it properly
                    3. Which recycling bin it belongs to
                    4. Any special preparation needed (e.g., rinse, remove labels)
                    
                    Please format your response clearly for each item.Don't make it bold.Show questions everytime along with answers.
                """.trimIndent()

                val response = callGeminiAPI(base64Image, prompt)

                withContext(Dispatchers.Main) {
                    btnGemini.text = "♻️ Get Recycling Tips"
                    btnGemini.isEnabled = true

                    // Open full screen activity with Gemini results
                    val intent = Intent(this@MainActivity, GeminiResultsActivity::class.java)
                    intent.putExtra("GEMINI_RESPONSE", response)
                    intent.putExtra("DETECTED_OBJECTS", objectsList)
                    startActivity(intent)
                }

            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    btnGemini.text = "♻️ Get Recycling Tips"
                    btnGemini.isEnabled = true
                    Toast.makeText(
                        this@MainActivity,
                        "Error: ${e.message}",
                        Toast.LENGTH_LONG
                    ).show()
                    e.printStackTrace()
                }
            }
        }
    }

    private fun bitmapToBase64(bitmap: Bitmap): String {
        val outputStream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 80, outputStream)
        val byteArray = outputStream.toByteArray()
        return Base64.encodeToString(byteArray, Base64.NO_WRAP)
    }

    private fun callGeminiAPI(base64Image: String, prompt: String): String {
        // Using Gemini 2.0 Flash - latest model with vision support
        val url = URL("https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key=$GEMINI_API_KEY")
        val connection = url.openConnection() as HttpURLConnection

        connection.requestMethod = "POST"
        connection.setRequestProperty("Content-Type", "application/json")
        connection.doOutput = true

        val jsonBody = JSONObject().apply {
            put("contents", JSONArray().apply {
                put(JSONObject().apply {
                    put("parts", JSONArray().apply {
                        put(JSONObject().apply {
                            put("text", prompt)
                        })
                        put(JSONObject().apply {
                            put("inline_data", JSONObject().apply {
                                put("mime_type", "image/jpeg")
                                put("data", base64Image)
                            })
                        })
                    })
                })
            })
        }

        connection.outputStream.use { outputStream ->
            val bytes = jsonBody.toString().toByteArray(Charsets.UTF_8)
            outputStream.write(bytes)
        }

        val responseCode = connection.responseCode
        if (responseCode == HttpURLConnection.HTTP_OK) {
            val response = connection.inputStream.bufferedReader(Charsets.UTF_8).use { reader ->
                reader.readText()
            }
            val jsonResponse = JSONObject(response)
            val candidates = jsonResponse.getJSONArray("candidates")
            val content = candidates.getJSONObject(0).getJSONObject("content")
            val parts = content.getJSONArray("parts")
            return parts.getJSONObject(0).getString("text")
        } else {
            val errorStream = connection.errorStream?.bufferedReader(Charsets.UTF_8)?.use { reader ->
                reader.readText()
            }
            throw Exception("API Error: $responseCode - $errorStream")
        }
    }

    private fun displayGeminiResults(geminiResponse: String) {
        val currentText = tvResults.text.toString()
        val updatedText = currentText.replace("🤖 Asking Gemini AI...\n", "")

        val finalText = buildString {
            append(updatedText)
            append("\n\n")
            append("━━━━━━━━━━━━━━━━━━━━\n")
            append("♻️ GEMINI RECYCLING GUIDE\n")
            append("━━━━━━━━━━━━━━━━━━━━\n\n")
            append(geminiResponse)
        }

        tvResults.text = finalText
        tvResults.setTextColor(Color.parseColor("#10B981"))

        findViewById<android.widget.ScrollView>(R.id.scrollResults).post {
            findViewById<android.widget.ScrollView>(R.id.scrollResults).fullScroll(android.view.View.FOCUS_DOWN)
        }
    }

    private fun checkCameraPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun requestCameraPermission() {
        permissionLauncher.launch(Manifest.permission.CAMERA)
    }

    private fun launchCamera() {
        val photoFile = File(cacheDir, "photo_${System.currentTimeMillis()}.jpg")
        currentPhotoUri = FileProvider.getUriForFile(
            this,
            "${packageName}.provider",
            photoFile
        )
        cameraLauncher.launch(currentPhotoUri)
    }

    override fun onDestroy() {
        super.onDestroy()
        if (::ortSession.isInitialized) ortSession.close()
        if (::ortEnvironment.isInitialized) ortEnvironment.close()
    }
}

data class Detection(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val confidence: Float,
    val classIndex: Int,
    val className: String
)