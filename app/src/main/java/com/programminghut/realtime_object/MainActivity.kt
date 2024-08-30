package com.programminghut.realtime_object

import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.*
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.view.Surface
import android.view.TextureView
import android.widget.ImageView
import androidx.core.content.ContextCompat
import com.programminghut.realtime_object.ml.SsdMobilenetV11Metadata1
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.view.WindowManager

class MainActivity : AppCompatActivity(), SensorEventListener {

    data class DetectedObject(
        var rect: RectF,
        val label: String,
        val score: Float,
        var initialAzimuth: Float,
        var initialPitch: Float
    )

    private val detectedObjects = mutableListOf<DetectedObject>()
    private var currentAzimuth: Float = 0f
    private var currentPitch: Float = 0f

    lateinit var labels:List<String>
    var colors = listOf<Int>(
        Color.BLUE, Color.GREEN, Color.RED, Color.CYAN, Color.GRAY, Color.BLACK,
        Color.DKGRAY, Color.MAGENTA, Color.YELLOW, Color.RED)
    val paint = Paint()
    lateinit var imageProcessor: ImageProcessor
    lateinit var bitmap:Bitmap
    lateinit var imageView: ImageView
    lateinit var cameraDevice: CameraDevice
    lateinit var handler: Handler
    lateinit var cameraManager: CameraManager
    lateinit var textureView: TextureView
    lateinit var model:SsdMobilenetV11Metadata1

    private lateinit var sensorManager: SensorManager
    private val rotationMatrix = FloatArray(9)
    private val orientationAngles = FloatArray(3)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        get_permission()

        labels = FileUtil.loadLabels(this, "labels.txt")
        imageProcessor = ImageProcessor.Builder().add(ResizeOp(300, 300, ResizeOp.ResizeMethod.BILINEAR)).build()
        model = SsdMobilenetV11Metadata1.newInstance(this)
        val handlerThread = HandlerThread("videoThread")
        handlerThread.start()
        handler = Handler(handlerThread.looper)

        imageView = findViewById(R.id.imageView)

        textureView = findViewById(R.id.textureView)
        textureView.surfaceTextureListener = object:TextureView.SurfaceTextureListener{
            override fun onSurfaceTextureAvailable(p0: SurfaceTexture, p1: Int, p2: Int) {
                open_camera()
            }
            override fun onSurfaceTextureSizeChanged(p0: SurfaceTexture, p1: Int, p2: Int) {
            }

            override fun onSurfaceTextureDestroyed(p0: SurfaceTexture): Boolean {
                return false
            }

            override fun onSurfaceTextureUpdated(p0: SurfaceTexture) {
                bitmap = textureView.bitmap!!
                var image = TensorImage.fromBitmap(bitmap)
                image = imageProcessor.process(image)

                val outputs = model.process(image)
                val locations = outputs.locationsAsTensorBuffer.floatArray
                val classes = outputs.classesAsTensorBuffer.floatArray
                val scores = outputs.scoresAsTensorBuffer.floatArray

                detectedObjects.clear() // Clear previous detections

                var mutable = bitmap.copy(Bitmap.Config.ARGB_8888, true)
                val canvas = Canvas(mutable)

                val h = mutable.height
                val w = mutable.width
                paint.textSize = h/15f
                paint.strokeWidth = h/85f
                var x = 0
                scores.forEachIndexed { index, fl ->
                    x = index
                    x *= 4
                    if(fl > 0.5){
                        val rect = RectF(locations.get(x+1)*w, locations.get(x)*h, locations.get(x+3)*w, locations.get(x+2)*h)
                        detectedObjects.add(DetectedObject(rect, labels.get(classes.get(index).toInt()), fl, currentAzimuth, currentPitch))
                    }
                }

                drawDetectedObjects(canvas)

                imageView.setImageBitmap(mutable)
            }
        }

        cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager

        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }

    private fun drawDetectedObjects(canvas: Canvas) {
        detectedObjects.forEachIndexed { index, obj ->
            paint.setColor(colors[index % colors.size])
            paint.style = Paint.Style.STROKE

            val adjustedRect = adjustRectPosition(obj)
            canvas.drawRect(adjustedRect, paint)

            paint.style = Paint.Style.FILL
            canvas.drawText("${obj.label} ${obj.score}", adjustedRect.left, adjustedRect.top, paint)
        }
    }

    private fun adjustRectPosition(obj: DetectedObject): RectF {
        val deltaAzimuth = currentAzimuth - obj.initialAzimuth
        val deltaPitch = currentPitch - obj.initialPitch

        // You may need to adjust these factors to get the desired effect
        val adjustX = deltaAzimuth * obj.rect.width() * 0.1f
        val adjustY = deltaPitch * obj.rect.height() * 0.1f

        return RectF(
            obj.rect.left - adjustX,
            obj.rect.top - adjustY,
            obj.rect.right - adjustX,
            obj.rect.bottom - adjustY
        )
    }

    override fun onDestroy() {
        super.onDestroy()
        model.close()
    }

    @SuppressLint("MissingPermission")
    fun open_camera(){
        cameraManager.openCamera(cameraManager.cameraIdList[0], object:CameraDevice.StateCallback(){
            override fun onOpened(p0: CameraDevice) {
                cameraDevice = p0

                var surfaceTexture = textureView.surfaceTexture
                var surface = Surface(surfaceTexture)

                var captureRequest = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
                captureRequest.addTarget(surface)

                cameraDevice.createCaptureSession(listOf(surface), object: CameraCaptureSession.StateCallback(){
                    override fun onConfigured(p0: CameraCaptureSession) {
                        p0.setRepeatingRequest(captureRequest.build(), null, null)
                    }
                    override fun onConfigureFailed(p0: CameraCaptureSession) {
                    }
                }, handler)
            }

            override fun onDisconnected(p0: CameraDevice) {
            }

            override fun onError(p0: CameraDevice, p1: Int) {
            }
        }, handler)
    }

    fun get_permission(){
        if(ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED){
            requestPermissions(arrayOf(android.Manifest.permission.CAMERA), 101)
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if(grantResults[0] != PackageManager.PERMISSION_GRANTED){
            get_permission()
        }
    }

    override fun onResume() {
        super.onResume()
        sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)?.also { rotationSensor ->
            sensorManager.registerListener(
                this,
                rotationSensor,
                SensorManager.SENSOR_DELAY_NORMAL
            )
        }
    }

    override fun onPause() {
        super.onPause()
        sensorManager.unregisterListener(this)
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // Do nothing
    }

    override fun onSensorChanged(event: SensorEvent) {
        if (event.sensor.type == Sensor.TYPE_ROTATION_VECTOR) {
            SensorManager.getRotationMatrixFromVector(rotationMatrix, event.values)
            SensorManager.getOrientation(rotationMatrix, orientationAngles)

            currentAzimuth = Math.toDegrees(orientationAngles[0].toDouble()).toFloat()
            currentPitch = Math.toDegrees(orientationAngles[1].toDouble()).toFloat()

            // Request a redraw of the image view
            imageView.invalidate()
        }
    }
}