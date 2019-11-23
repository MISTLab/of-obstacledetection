using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.IO;

//
//        8888888          d8b 888    d8b          888 d8b                   888    d8b                   
//          888            Y8P 888    Y8P          888 Y8P                   888    Y8P                   
//          888                888                 888                       888                          
//          888   88888b.  888 888888 888  8888b.  888 888 .d8888b   8888b.  888888 888  .d88b.  88888b.  
//          888   888 "88b 888 888    888     "88b 888 888 88K          "88b 888    888 d88""88b 888 "88b 
//          888   888  888 888 888    888 .d888888 888 888 "Y8888b. .d888888 888    888 888  888 888  888 
//          888   888  888 888 Y88b.  888 888  888 888 888      X88 888  888 Y88b.  888 Y88..88P 888  888 
//        8888888 888  888 888  "Y888 888 "Y888888 888 888  88888P' "Y888888  "Y888 888  "Y88P"  888  888 
//                                                                                                                                                                                                           

public class Unit : MonoBehaviour {
    [Header("Attached Objects")]
    public Transform astar;
    public Transform target;
    public Camera cam;
    public Material ObstacleMat;
    public Material CrashObjMat;

    [Header("Data")]
    public int dataCounter = 0;
    public int currentObstacleID = 0;
    public int maxEpochs = 1;
    public bool fixedObstacle = false;

    [Header("Agent Settings")]
    public float turnSpeed = 4f;
    public float turnDst = 0.2f;
    public float stoppingDst = 0f;
    public Vector3 colliderSize = new Vector3(0.2f, 0.2f, 0.4f);

    [Header("Capture Settings")]
    public string outputFolder;
    public float randomRotationDeg = 0.25f;
    public float captureDistance = 0.05f;
    public int sideObstacleCount = 30;
    public int imageWidth = 640;
    public int imageHeight = 480;
    public float crashChance = 0.5f;
    public float crashDistance = 2f;
    public int collectEveryXFrames = 3;
    public Vector2 fovMinMax = new Vector2(80f, 100f);
    public float maxPitchOffset = 10f;
    public float maxOFAngleOffset = 30f;

    private Path path;
    private Grid grid;
    private ImageSynthesis flowCam;
    private GameObject[] loadedObstacles;
    private List<GameObject> obstaclesList = new List<GameObject>();
    private GameObject currentObstacle;
    private List<CollectedData> data = new List<CollectedData>();
    private CollectedData lastData;
    private bool goUp = true;
    private bool crashed = false;
    private bool showCollision = false;
    private float flowCamSensitivity;
    private float pitchoffset = 5f;
    private bool isSteeringData = false;
    private float cumAng = 0;
    private int currentID = 0;
    private int currentEpoch = 0;
    private int counter = 0;

    // structure for recording data
    struct CollectedData {
        public int ID;
        public string imgName;
        public int timeStamp;
        public float steeringAngle;
        public int collisionData;
        public Vector3 position;
        public Vector3 direction;
        public int dataType;
        public int obstID;
        public int epoch;

        public CollectedData(int i, string name, int ts, float sa, int cd, Vector3 pos, Vector3 dir, int type, int oi, int ep) {
            ID = i;
            imgName = name;
            timeStamp = ts;
            steeringAngle = sa;
            collisionData = cd;
            position = pos;
            direction = dir;
            dataType = type;
            obstID = oi;
            epoch = ep;
        }
    }

    /// <summary>
    /// Start the simulator and build the environment.
    /// </summary>
    private void Start() {
        // fixed framerate to equalize differences in performance
        UnityEngine.QualitySettings.vSyncCount = 0;
        UnityEngine.Application.targetFrameRate = 30;

        // init variables
        grid = FindObjectOfType<Grid>();
        flowCam = cam.GetComponent<ImageSynthesis>();
        flowCamSensitivity = flowCam.opticalFlowSensitivity;
        if (!fixedObstacle) currentObstacleID = 0;

        lastData = new CollectedData(-1, "init", 0, 0, 0, Vector3.zero, Vector3.zero, 0, currentObstacleID, currentEpoch);

        // load obstacles
        loadedObstacles = Resources.LoadAll<GameObject>("Ready");

        // IMPORTANT: RUN THESE LINES ONLY WITH NEW MODELS ONCE!!!
        // otherwise new meshcolliders will be added to the prefabs all the time 
        //foreach (GameObject obj in loadedObstacles)
        //{
        //    MeshCollider mc = obj.AddComponent<MeshCollider>();
        //    mc.sharedMesh = obj.GetComponentInChildren<MeshFilter>().sharedMesh;
        //}

        // generate environment
        GenerateEnvironment();

        // create output path
        if (Directory.Exists(outputFolder)) { Directory.Delete(outputFolder, true); }
        Directory.CreateDirectory(outputFolder);

        // request new path & start moving
        PathRequestManager.RequestPath(new PathRequest(transform.position, target.position, OnPathFound));
}

    //
    //        8888888b.           888                   .d8888b.                                             888                    
    //        888  "Y88b          888                  d88P  Y88b                                            888                    
    //        888    888          888                  888    888                                            888                    
    //        888    888  8888b.  888888  8888b.       888         .d88b.  88888b.   .d88b.  888d888 8888b.  888888 .d88b.  888d888 
    //        888    888     "88b 888        "88b      888  88888 d8P  Y8b 888 "88b d8P  Y8b 888P"      "88b 888   d88""88b 888P"   
    //        888    888 .d888888 888    .d888888      888    888 88888888 888  888 88888888 888    .d888888 888   888  888 888     
    //        888  .d88P 888  888 Y88b.  888  888      Y88b  d88P Y8b.     888  888 Y8b.     888    888  888 Y88b. Y88..88P 888     
    //        8888888P"  "Y888888  "Y888 "Y888888       "Y8888P88  "Y8888  888  888  "Y8888  888    "Y888888  "Y888 "Y88P"  888     
    //                                                                                                                              


    /// <summary>
    /// Detects collisions with all types of obstacles.
    /// </summary>
    /// <returns><c>true</c>, if collision is imminent, <c>false</c> otherwise.</returns>
    private bool DetectCollision() {
        // bitshifted int to mask both layers of obstacles
        int X = 1 << 8;
        X += 1 << 9;

        // check for actual crash with both layers 
        Vector3 pos = transform.position + transform.forward * colliderSize.z / 2f;
        crashed = Physics.CheckBox(pos, colliderSize, transform.localRotation, X);

        // check for possible collision with a spherical ray
        RaycastHit hitInfo;
        bool recordEverything = false;
        bool aboutToCrash = Physics.SphereCast(transform.position,
                                               colliderSize.x * 2f,
                                               transform.forward,
                                               out hitInfo, crashDistance, X);

        // make sure the distance to the obstacle is within range
        if (aboutToCrash) {
            Collider obstCollider = currentObstacle.GetComponent<Collider>();
            float distToObstacle = Vector3.Distance(transform.position, 
                                                    obstCollider.ClosestPointOnBounds(transform.position));

            if (distToObstacle <= crashDistance) {
                lastData.collisionData = 1;
                recordEverything = true;
                showCollision = true;
            }
        }
        return recordEverything;
    }


    /// <summary>
    /// Collects the current state of the agent regarding collisions and steering angles.
    /// </summary>
    private void CollectData(float currentSteeringAngle) {
        // add small rotation to camera for more randomization
        float rr = randomRotationDeg;
        cam.transform.localRotation = Quaternion.Euler(Random.Range(-rr, rr) + pitchoffset,
                                                       Random.Range(-rr, rr),
                                                       Random.Range(-rr, rr));

        // capture flow image
        string imgName = dataCounter.ToString("0000000");
        flowCam.Save(imgName, imageWidth, imageHeight, outputFolder);

        // add last dataset with current steering angle and crash data
        lastData.steeringAngle = currentSteeringAngle;
        data.Add(lastData);

        // create new dataset for next step
        CollectedData newData = new CollectedData(currentID, imgName, dataCounter, 0,
                                                crashed ? 1 : 0,
                                                transform.position,
                                                transform.rotation.eulerAngles,
                                                isSteeringData ? 1 : 0,
                                                currentObstacleID,
                                                currentEpoch);

        // update last data
        lastData = newData;
        dataCounter++;
    }


    /// <summary>
    /// Generates the environment for the experiments.
    /// </summary>
    private void GenerateEnvironment() {

        // delete old obstacles
        if (obstaclesList.Count > 0) {
            foreach (GameObject obj in obstaclesList) {
                obj.SetActive(false);
                Destroy(obj.gameObject);
            }
            obstaclesList.Clear();
        }

        // set new track width
        grid.gridWorldSize = new Vector2(Random.Range(12, 20), grid.gridWorldSize.y);
        Vector2 gS = grid.gridWorldSize;

        // place detectable obstacles on the sidelines
        for (int i = 0; i < sideObstacleCount; i++) {
            
            // calculate coordinates
            float x = RandomFromDistribution.RandomRangeNormalDistribution(-3f, 4f, 
                      RandomFromDistribution.ConfidenceLevel_e._99) + gS.x / 2;
            float z = Mathf.Lerp(-gS.y/2, gS.y/2, (float)i / (float)sideObstacleCount);

            // objects on the right
            GameObject obsToBuild = loadedObstacles[Random.Range(0, loadedObstacles.Length)];
            Vector3 obsSize = obsToBuild.GetComponentInChildren<Renderer>().bounds.size;
            float maxSize = Mathf.Max(obsSize.x, obsSize.y);

            Vector3 newPos = new Vector3(x + maxSize / 3, 0f, z);
            Quaternion newRot = Quaternion.Euler(Random.Range(-5f, 5f), 
                                                 Random.Range(0f, 360f), 
                                                 Random.Range(-5f, 5f));

            GameObject newObs = Instantiate(obsToBuild, newPos, newRot) as GameObject;

            newObs.GetComponentInChildren<MeshRenderer>().material = ObstacleMat;
            newObs.layer = 8;
            obstaclesList.Add(newObs);

            // objects on the left
            obsToBuild = loadedObstacles[Random.Range(0, loadedObstacles.Length)];
            obsSize = obsToBuild.GetComponentInChildren<Renderer>().bounds.size;
            maxSize = Mathf.Max(obsSize.x, obsSize.y);

            newPos = new Vector3(-x  - maxSize / 3, 0f, z);
            newRot = Quaternion.Euler(Random.Range(-5f, 5f), 
                                      Random.Range(0f, 360f), 
                                      Random.Range(-5f, 5f));

            newObs = Instantiate(obsToBuild, newPos, newRot) as GameObject;
            newObs.GetComponentInChildren<MeshRenderer>().material = ObstacleMat;
            newObs.layer = 8;
            obstaclesList.Add(newObs);
        }

        // update current obstacle ID if not set to fixed
        if (!fixedObstacle) {
            if (currentObstacleID < loadedObstacles.Length - 1)
                currentObstacleID++;
            else {
                Debug.Log("Epoch: " + currentEpoch++);
                currentObstacleID = 0;
            }
        }

        // load central obstacle to avoid
        GameObject otb = loadedObstacles[currentObstacleID];
        Vector3 oS = otb.GetComponentInChildren<Renderer>().bounds.size;

        // make sure it's not too big
        float mS = Mathf.Max(oS.x, oS.y);
        while (mS > grid.gridWorldSize.x / 2f) {
            otb = loadedObstacles[Random.Range(0, loadedObstacles.Length)];
            oS = otb.GetComponentInChildren<Renderer>().bounds.size;
            mS = Mathf.Max(oS.x, oS.y);
        }

        // place it
        Vector3 nP = new Vector3(Random.value - 0.5f, Random.value - 0.25f, 0f);
        Quaternion nR = Quaternion.Euler(Random.Range(-5f, 5f), Random.Range(0f, 360f), Random.Range(-5f, 5f));
        currentObstacle = Instantiate(otb, nP, nR) as GameObject;        
        obstaclesList.Add(currentObstacle);

        // make obstacle undetectable in x % of the cases to force crashes
        if (Random.value < crashChance) {
            currentObstacle.layer = 9;
            currentObstacle.GetComponentInChildren<MeshRenderer>().material = CrashObjMat;
            isSteeringData = false;
        }
        else {
            currentObstacle.layer = 8;
            currentObstacle.GetComponentInChildren<MeshRenderer>().material = ObstacleMat;
            isSteeringData = true;
        }

        grid.Awake();
    }


//
//        888b     d888                                                            888    
//        8888b   d8888                                                            888    
//        88888b.d88888                                                            888    
//        888Y88888P888  .d88b.  888  888  .d88b.  88888b.d88b.   .d88b.  88888b.  888888 
//        888 Y888P 888 d88""88b 888  888 d8P  Y8b 888 "888 "88b d8P  Y8b 888 "88b 888    
//        888  Y8P  888 888  888 Y88  88P 88888888 888  888  888 88888888 888  888 888    
//        888   "   888 Y88..88P  Y8bd8P  Y8b.     888  888  888 Y8b.     888  888 Y88b.  
//        888       888  "Y88P"    Y88P    "Y8888  888  888  888  "Y8888  888  888  "Y888 
//                                                                                    


    /// <summary>
    /// Called when a path requested, starts the movement or requests a new environment if path is blocked.
    /// </summary>
    public void OnPathFound(Vector3[] waypoints, bool pathSuccessful) {
        crashed = false;
        if (pathSuccessful) {
            path = new Path(waypoints, transform.position, turnDst, stoppingDst);
           
            StopCoroutine("FollowPath"); 
            StartCoroutine("FollowPath");
        } else {
            GenerateEnvironment();
            PathRequestManager.RequestPath(new PathRequest(transform.position, target.position, OnPathFound));
        }
	}


    /// <summary>
    /// Moves the agent along the path.
    /// Also contains the mechanics for recording the data and initiating a new experiment.
    /// </summary>
    private IEnumerator FollowPath() {

        bool followingPath = true;
        int pathIndex = 0;
        transform.LookAt(path.lookPoints[0]);

        while (followingPath)
        {
            Vector2 pos2D = new Vector2(transform.position.x, transform.position.z);
            while (path.turnBoundaries[pathIndex].HasCrossedLine(pos2D))
            {
                if (pathIndex == path.finishLineIndex)
                {
                    followingPath = false;
                    break;
                }
                pathIndex++;
            }

            // rotate & move
            Quaternion targetRotation = Quaternion.LookRotation(path.lookPoints[pathIndex] - transform.position);
            transform.rotation = Quaternion.Lerp(transform.rotation, targetRotation, 0.1f);
            float captrueDistanceWithVar = captureDistance + Random.Range(-captureDistance + 0.01f, captureDistance);
            transform.Translate(Vector3.forward * captrueDistanceWithVar, Space.Self);

            // calculate steering angle
            float currentSteeringAngle = GetAngleDifference(transform.rotation.eulerAngles.y,
                                                            lastData.direction.y);

            // adjust sensitivity of optical flow camera when rotating
            flowCam.opticalFlowSensitivity = flowCamSensitivity / (captureDistance * 4)
                                             - Mathf.Clamp(Mathf.Abs(currentSteeringAngle * 27f),
                                             0f, flowCamSensitivity / (captureDistance * 4) - 7f);


            // set data for next steps testing even if no new data was added
            lastData.direction = transform.rotation.eulerAngles;
            lastData.position = transform.position;

            // data collection logic: 
            // record every Xth frame when no collision imminent
            // record every frame when collision is happening
            // don't record data when crash occured until reset
            if (ShouldCollectData(DetectCollision())) CollectData(currentSteeringAngle);

            // reset after ostacle was passed midway or after turning back from avoidance or crash occured
            if ((goUp && transform.position.z > -0.5f)
            || (!goUp && transform.position.z < 0.5f)
            || (Mathf.Abs(cumAng + currentSteeringAngle) + 1f < Mathf.Abs(cumAng))
            || crashed) {
                SwitchPositions();
            } else {
                cumAng += currentSteeringAngle;
            }
            counter++;

            yield return null;
        }
	}


    /// <summary>
    /// Switches the positions of target and seeker. 
    /// Every other step a new environment is generated.
    /// </summary>
    private void SwitchPositions()
    {
        // stop program if max epochs reached
        if (currentEpoch >= maxEpochs) {
            #if UNITY_EDITOR
                UnityEditor.EditorApplication.isPlaying = false;
            #else
                Application.Quit();
            #endif
        }

        // new height
        float h = 0.5f + Random.value * 1.5f;

        // new main camera fov angle
        Camera.main.fieldOfView = Random.Range(fovMinMax.x, fovMinMax.y);

        // new pitch offset
        pitchoffset = Random.Range(-maxPitchOffset, maxPitchOffset);

        // new optical flow angle offset
        flowCam.opticalFlowAngleOffset = Random.Range(-maxOFAngleOffset, maxOFAngleOffset) * Mathf.PI / 180;

        // check if seeker is supposed to go up or down
        if (goUp) {
            float rr = 360;
            currentObstacle.transform.localRotation = Quaternion.Euler(0f, Random.Range(-rr, rr), 0f);
            target.transform.position = new Vector3(0, h, -grid.gridWorldSize.y / 2 + 1);
            transform.position = new Vector3(0, h, grid.gridWorldSize.y / 2 - 1);
        } else {
            GenerateEnvironment();
            target.transform.position = new Vector3(0, h, grid.gridWorldSize.y / 2 - 1);
            transform.position = new Vector3(0, h, -grid.gridWorldSize.y / 2 + 1);
        }

        // set new height to grid
        astar.position = new Vector3(0, h, 0);
        transform.LookAt(target.transform);
        lastData.direction = transform.rotation.eulerAngles;
        goUp = !goUp;

        // reset cumulative angle and set next ID
        cumAng = 0;
        currentID++;
        crashed = false;
        showCollision = false;

        // request new path
        PathRequestManager.RequestPath(new PathRequest(transform.position, target.position, OnPathFound));
    }


    //
    //        888    888          888                                888b     d888          888    888                    888          
    //        888    888          888                                8888b   d8888          888    888                    888          
    //        888    888          888                                88888b.d88888          888    888                    888          
    //        8888888888  .d88b.  888 88888b.   .d88b.  888d888      888Y88888P888  .d88b.  888888 88888b.   .d88b.   .d88888 .d8888b  
    //        888    888 d8P  Y8b 888 888 "88b d8P  Y8b 888P"        888 Y888P 888 d8P  Y8b 888    888 "88b d88""88b d88" 888 88K      
    //        888    888 88888888 888 888  888 88888888 888          888  Y8P  888 88888888 888    888  888 888  888 888  888 "Y8888b. 
    //        888    888 Y8b.     888 888 d88P Y8b.     888          888   "   888 Y8b.     Y88b.  888  888 Y88..88P Y88b 888      X88 
    //        888    888  "Y8888  888 88888P"   "Y8888  888          888       888  "Y8888   "Y888 888  888  "Y88P"   "Y88888  88888P' 
    //                                888                                                                                              
    //                                888                                                                                              
    //                                888    
    //


    /// <summary>
    /// Decides if data should be collected.
    /// </summary>
    private bool ShouldCollectData(bool recordEverything) {
        if (lastData.collisionData == 0 || !crashed) {
            if (!recordEverything && counter % collectEveryXFrames != 0)
                return false;
            return true;
        }
        return false;
    }
    

    /// <summary>
    /// When quitting the simulation, write labels.txt.
    /// </summary>
    private void OnApplicationQuit()
    {
        data.Add(lastData);
        var fname = System.IO.Path.Combine(outputFolder, "labels.txt");

        StreamWriter writer = File.CreateText(fname);
        writer.WriteLine("ID; Filename; X; Y; Z; Experiment Type; Steering Angle; Collision Data; Obstacle ID; Epoch");
        foreach (CollectedData current in data)
        {
            writer.WriteLine(current.ID + ";" + 
                             current.imgName + ".png;" +
                             current.position.x + ";" +
                             current.position.y + ";" +
                             current.position.z + ";" +
                             current.dataType + ";" +
                             Round(current.steeringAngle, 4) + ";" +
                             current.collisionData + ";" +
                             current.obstID + ";" +
                             current.epoch);
        }
        writer.Close();
    }


    /// <summary>
    /// Calculates the difference between two angles.
    /// </summary>
    public static float GetAngleDifference(float a1, float a2) {
        float d = (a1 - a2) % 360;
        d += (d > 180) ? - 360 : (d < -180) ? 360 : 0;
        return d;
    }


    /// <summary>
    /// Round to digit x.
    /// </summary>
    public static float Round(float value, int digits)
    {
        float mult = Mathf.Pow(10.0f, digits);
        return Mathf.Round(value * mult) / mult;
    }


    /// <summary>
    /// Draws stuff for debugging.
    /// </summary>
    public void OnDrawGizmos() {
        if (path != null) {
            path.DrawWithGizmos();

            Gizmos.color = showCollision ? Color.red : Color.white;
            Gizmos.DrawSphere(transform.position + transform.up * 3f, 1f);
        }
        Gizmos.color = Color.green;
        Gizmos.DrawSphere(target.position, 1f);
    }
}
