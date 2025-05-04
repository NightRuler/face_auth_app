'use client';

import { useEffect, useRef, useState } from 'react';
import {
  FaceLandmarker,
  FilesetResolver,
  DrawingUtils,
} from '@mediapipe/tasks-vision';

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [faceLandmarker, setFaceLandmarker] = useState<FaceLandmarker | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [similarity, setSimilarity] = useState<number | null>(null);
  const threshold = 0.9;

  useEffect(() => {
    async function loadModel() {
      const filesetResolver = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm'
      );

      const landmarkDetector = await FaceLandmarker.createFromOptions(filesetResolver, {
        baseOptions: {
          modelAssetPath:
            'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
          delegate: 'GPU',
        },
        outputFaceBlendshapes: false,
        runningMode: 'VIDEO',
        numFaces: 1,
      });

      setFaceLandmarker(landmarkDetector);
    }

    loadModel();
  }, []);

  const startCamera = async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      alert('Camera not supported');
      return;
    }

    const stream = await navigator.mediaDevices.getUserMedia({ video: true });

    if (videoRef.current) {
      videoRef.current.srcObject = stream;

      videoRef.current.onloadeddata = () => {
        videoRef.current?.play();

        const waitUntilReady = setInterval(() => {
          if (
            videoRef.current?.videoWidth &&
            videoRef.current?.videoHeight
          ) {
            clearInterval(waitUntilReady);
            setIsRunning(true);
            requestAnimationFrame(runDetection);
          }
        }, 100);
      };
    }
  };

  const runDetection = async () => {
    if (
      !faceLandmarker ||
      !videoRef.current ||
      !canvasRef.current ||
      !videoRef.current.videoWidth ||
      !videoRef.current.videoHeight
    ) {
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const now = performance.now();

    const result = await faceLandmarker.detectForVideo(video, now);
    const landmarks = result.faceLandmarks?.[0];

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    if (landmarks && ctx) {
      const drawer = new DrawingUtils(ctx);
      drawer.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_TESSELATION,
        { color: '#00FF00', lineWidth: 1 }
      );
    }

    if (isRunning) {
      requestAnimationFrame(runDetection);
    }
  };

  const flattenLandmarks = (landmarks: any[]) =>
    landmarks.flatMap((p) => [p.x, p.y, p.z]);

  const cosineSimilarity = (a: number[], b: number[]) => {
    const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
    const magA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
    const magB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
    return dot / (magA * magB);
  };

  const registerFace = async () => {
    if (!faceLandmarker || !videoRef.current) return;

    const now = performance.now();
    const result = await faceLandmarker.detectForVideo(videoRef.current, now);
    const landmarks = result.faceLandmarks?.[0];

    if (landmarks) {
      const vector = flattenLandmarks(landmarks);
      localStorage.setItem('registeredFaceVector', JSON.stringify(vector));
      alert('✅ Face registered and saved!');
    } else {
      alert('❌ No face detected. Try again.');
    }
  };

  const authenticateFace = async () => {
    if (!faceLandmarker || !videoRef.current) return;

    const saved = localStorage.getItem('registeredFaceVector');
    if (!saved) {
      alert('❌ No registered face found. Register first.');
      return;
    }

    const storedVector = JSON.parse(saved) as number[];

    const now = performance.now();
    const result = await faceLandmarker.detectForVideo(videoRef.current, now);
    const landmarks = result.faceLandmarks?.[0];

    if (!landmarks) {
      alert('❌ No face detected for authentication.');
      return;
    }

    const currentVector = flattenLandmarks(landmarks);
    const sim = cosineSimilarity(storedVector, currentVector);
    setSimilarity(sim);
  };

  return (
    <div style={{ padding: '2rem' }}>
      <h1>🧠 Face Authentication Demo</h1>

      <div style={{ position: 'relative', width: 480 }}>
        <video
          ref={videoRef}
          style={{ width: '100%', borderRadius: 8 }}
          muted
          playsInline
        />
        <canvas
          ref={canvasRef}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: 'auto',
          }}
        />
      </div>

      <div style={{ marginTop: '1rem' }}>
        <button onClick={startCamera} style={{ marginRight: 10 }}>
          📷 Start Camera
        </button>
        <button onClick={registerFace} style={{ marginRight: 10 }}>
          🔐 Register Face
        </button>
        <button onClick={authenticateFace}>✅ Authenticate Face</button>
      </div>

      {similarity !== null && (
        <div style={{ marginTop: '1rem' }}>
          <p>🔎 Cosine Similarity: {similarity.toFixed(4)}</p>
          <p>
            {similarity > threshold ? '✅ Face Authenticated' : '❌ Authentication Failed'}
          </p>
        </div>
      )}
    </div>
  );
}
