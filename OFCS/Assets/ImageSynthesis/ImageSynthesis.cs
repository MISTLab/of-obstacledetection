using UnityEngine;
using UnityEngine.Rendering;
using System.Collections;
using System.IO;

[RequireComponent(typeof(Camera))]
public class ImageSynthesis : MonoBehaviour
{

    // pass configuration
    private CapturePass[] capturePasses = {
        new CapturePass() { name = "img_" },
        new CapturePass() { name = "flow_", supportsAntialiasing = false, needsRescale = true },
    };

	struct CapturePass {
		// configuration
		public string name;
		public bool supportsAntialiasing;
		public bool needsRescale;
        public bool isRaw;
        public bool saveThis;
        public CapturePass(string name_) { name = name_; 
                                           supportsAntialiasing = true; 
                                           needsRescale = false; 
                                           camera = null; 
                                           isRaw = false;
                                           saveThis = false; }
        public CapturePass(string name_, bool sA_, bool nR_, bool iR_, bool sT_) { name = name_; 
            supportsAntialiasing = sA_; 
            needsRescale = nR_; 
            camera = null; 
            isRaw = iR_;
            saveThis = sT_; }

        // impl
        public Camera camera;
	}

	public Shader opticalFlowShader;
    public Shader opticalFlowShaderRaw;
    public Shader replacementShader;
    public float opticalFlowSensitivity;
    public float opticalFlowAngleOffset;
    public bool renderReal = false;
    public bool renderFlow = false;
    public bool renderRawFlow = false;
    public bool renderDepth = false;

	// cached materials
	private Material opticalFlowMaterial;
    private Material opticalFlowMaterialRaw;

    void Start()
	{
		// default fallbacks, if shaders are unspecified
		if (!opticalFlowShader)
			opticalFlowShader = Shader.Find("Hidden/OpticalFlow");
        if (!opticalFlowShaderRaw)
            opticalFlowShaderRaw = Shader.Find("Hidden/OpticalFlowRawUV");
        if (!replacementShader)
            replacementShader = Shader.Find("Hidden/UberReplacement");

        // save or render only wanted passes
        if (renderReal) capturePasses[0].saveThis = true;
        if (renderFlow) capturePasses[1].saveThis = true;
        if (renderRawFlow) {
            CapturePass[] np = AddPass(capturePasses, new CapturePass("flow_raw_", false, true, true, true));
            capturePasses = np;
        }
        if (renderDepth)
        {
            CapturePass[] np = AddPass(capturePasses, new CapturePass("depth_", true, false, false, true));
            capturePasses = np;
        }

        // use real camera to capture final image
        capturePasses[0].camera = GetComponent<Camera>();
		for (int q = 1; q < capturePasses.Length; q++)
			capturePasses[q].camera = CreateHiddenCamera(capturePasses[q].name);

		OnCameraChange();
	}

    static private CapturePass[] AddPass(CapturePass[] arr, CapturePass inp) {
        int n = arr.Length;
        CapturePass[] finalArray = new CapturePass[n + 1];
        for (int i = 0; i < n; i++)
            finalArray[i] = arr[i];
        finalArray[n] = inp;
        return finalArray;
    }

	void LateUpdate()
	{
		OnCameraChange();
	}

	private Camera CreateHiddenCamera(string cname)
	{
        var go = new GameObject(cname, typeof(Camera))
        {
            hideFlags = HideFlags.HideAndDontSave
        };
        go.transform.parent = transform;

		var newCamera = go.GetComponent<Camera>();
		return newCamera;
	}

	static private void SetupCameraWithPostShader(Camera cam, Material material, DepthTextureMode depthTextureMode = DepthTextureMode.None)
	{
		var cb = new CommandBuffer();
		cb.Blit(null, BuiltinRenderTextureType.CurrentActive, material);
		cam.AddCommandBuffer(CameraEvent.AfterEverything, cb);
		cam.depthTextureMode = depthTextureMode;
	}

    enum ReplacelementModes
    {
        ObjectId = 0,
        CatergoryId = 1,
        DepthCompressed = 2,
        DepthMultichannel = 3,
        Normals = 4
    };

    static private void SetupCameraWithReplacementShader(Camera cam, Shader shader, ReplacelementModes mode, Color clearColor)
    {
        var cb = new CommandBuffer();
        cb.SetGlobalFloat("_OutputMode", (int)mode); // @TODO: CommandBuffer is missing SetGlobalInt() method
        cam.AddCommandBuffer(CameraEvent.BeforeForwardOpaque, cb);
        cam.AddCommandBuffer(CameraEvent.BeforeFinalPass, cb);
        cam.SetReplacementShader(shader, "");
        cam.backgroundColor = clearColor;
        cam.clearFlags = CameraClearFlags.SolidColor;
    }

    public void OnCameraChange()
	{   

		int targetDisplay = 1;
        int rawFlowPass = -1;
        int depthPass = -1;

        var mainCamera = GetComponent<Camera>();
		for (int j = 0; j < capturePasses.Length; j++) {
			if (capturePasses[j].camera == mainCamera)
				continue;

            // cleanup capturing camera
            capturePasses[j].camera.RemoveAllCommandBuffers();

            // copy all "main" camera parameters into capturing camera
            capturePasses[j].camera.CopyFrom(mainCamera);

            // set targetDisplay here since it gets overriden by CopyFrom()
            capturePasses[j].camera.targetDisplay = targetDisplay++;

            // get indices of raw flow and depth
            if (capturePasses[j].name == "depth_")
                depthPass = j;
            if (capturePasses[j].name == "flow_raw_")
                rawFlowPass = j;
        }

        // cache materials and setup material properties
        if (!opticalFlowMaterial || opticalFlowMaterial.shader != opticalFlowShader)
			opticalFlowMaterial = new Material(opticalFlowShader);
		opticalFlowMaterial.SetFloat("_Sensitivity", opticalFlowSensitivity);
        opticalFlowMaterial.SetFloat("_Angle_Offset", opticalFlowAngleOffset);

        if (!opticalFlowMaterialRaw || opticalFlowMaterialRaw.shader != opticalFlowShaderRaw)
            opticalFlowMaterialRaw = new Material(opticalFlowShaderRaw);

        // setup command buffers and replacement shaders
        SetupCameraWithPostShader(capturePasses[1].camera, 
                                  opticalFlowMaterial,
                                  DepthTextureMode.Depth | DepthTextureMode.MotionVectors);
                                  
        if (renderRawFlow) 
            SetupCameraWithPostShader(capturePasses[rawFlowPass].camera, 
                                      opticalFlowMaterialRaw, 
                                      DepthTextureMode.Depth | DepthTextureMode.MotionVectors);

        if (renderDepth)
            SetupCameraWithReplacementShader(capturePasses[depthPass].camera,
                                             replacementShader,
                                             ReplacelementModes.DepthCompressed,
                                             Color.white);
    }

//
//         .d8888b.                    d8b
//        d88P  Y88b                   Y8P
//        Y88b.
//         "Y888b.    8888b.  888  888 888 88888b.   .d88b.
//            "Y88b.     "88b 888  888 888 888 "88b d88P"88b
//              "888 .d888888 Y88  88P 888 888  888 888  888
//        Y88b  d88P 888  888  Y8bd8P  888 888  888 Y88b 888
//         "Y8888P"  "Y888888   Y88P   888 888  888  "Y88888
//                                                       888
//                                                  Y8b d88P
//                                                   "Y88P"
//

    public void Save(string filename, int width = -1, int height = -1, string path = "")
	{
		if (width <= 0 || height <= 0) {
			width = Screen.width;
			height = Screen.height;
		}

	
		// execute as coroutine to wait for the EndOfFrame before starting capture
		StartCoroutine(WaitForEndOfFrameAndSave(filename, path, width, height));
	}

	private IEnumerator WaitForEndOfFrameAndSave(string filename, string path, int width, int height)
	{
		yield return new WaitForEndOfFrame();
		Save(filename, path, width, height);
	}

	private void Save(string filename, string path, int width, int height)
	{
        foreach (var pass in capturePasses){
            if (pass.saveThis) {
                string fname = pass.name + filename;
                var pathWithoutExtension = System.IO.Path.Combine(path, fname);
                Save(pass.camera,
                     pathWithoutExtension,
                     width, height,
                     pass.supportsAntialiasing,
                     pass.needsRescale,
                     pass.isRaw);
            }
        }
	}

	private void Save(Camera cam, string filename, int width, int height, bool supportsAntialiasing, bool needsRescale, bool isRaw)
	{
        string filenameExtension = ".png";

        // make raw texture half sized
        if (isRaw) {
            filenameExtension = ".bin";
        }

		var mainCamera = GetComponent<Camera>();
		var depth = 24;
		var format = RenderTextureFormat.ARGBFloat;
		var readWrite = RenderTextureReadWrite.Default;
		var antiAliasing = (supportsAntialiasing) ? Mathf.Max(1, QualitySettings.antiAliasing) : 1;

		var finalRT =
			RenderTexture.GetTemporary(width, height, depth, format, readWrite, antiAliasing);
		var renderRT = (!needsRescale) ? finalRT :
			RenderTexture.GetTemporary(mainCamera.pixelWidth, mainCamera.pixelHeight, depth, format, readWrite, antiAliasing);

        var prevActiveRT = RenderTexture.active;
        var prevCameraRT = cam.targetTexture;

        // render to offscreen texture (readonly from CPU side)
        RenderTexture.active = renderRT;
        cam.targetTexture = renderRT;
        cam.Render();

        if (needsRescale) {
            // blit to rescale (see issue with Motion Vectors in @KNOWN ISSUES)
            RenderTexture.active = finalRT;
            Graphics.Blit(renderRT, finalRT);
            RenderTexture.ReleaseTemporary(renderRT);
        }

        // read offsreen texture contents into the CPU readable texture
        Texture2D tex = new Texture2D(width, height, TextureFormat.RGBAFloat, false);
        tex.ReadPixels(new Rect(0, 0, tex.width, tex.height), 0, 0);
        tex.Apply();

        // encode texture into PNG or not if flow
        byte[] bytes = isRaw ? tex.GetRawTextureData() : tex.EncodeToPNG();
        File.WriteAllBytes(filename + filenameExtension, bytes);

        // restore state and cleanup
        cam.targetTexture = prevCameraRT;
		RenderTexture.active = prevActiveRT;

		Object.Destroy(tex);
		RenderTexture.ReleaseTemporary(finalRT);
	}
}
