import React, { useEffect, useRef, useState } from "react";
import { Button, Row, Col, message, Tour } from "antd";
import {
  UndoOutlined,
  ClearOutlined,
  QuestionOutlined,
} from "@ant-design/icons";
import CanvasDraw from "react-canvas-draw";
import pica from "pica";
import * as tf from "@tensorflow/tfjs";
import "./App.css";
import { useCallback } from "react";

function App() {
  const canvasRef = useRef(null);
  const [pred, setPred] = useState();
  const [prob, setProb] = useState();
  const [loadingPredict, setLoadingPredict] = useState(false);
  const picaInstance = pica();
  const [model, setModel] = useState();
  const results = useRef(null);
  const canvas = useRef(null);
  const clearButton = useRef(null);
  const undoButton = useRef(null);
  const predictButton = useRef(null);
  const [open, setOpen] = useState(false);

  // var model;
  const loadModel = useCallback(async () => {
    let m = await tf.loadLayersModel("model.json");
    setModel(m);
    // console.log(model);
  }, []);

  const saveDrawing = async () => {
    const canvas = canvasRef.current.canvas.drawing;
    // const context = canvas.getContext("2d");

    const smallCanvas = document.createElement("canvas");
    smallCanvas.width = 28;
    smallCanvas.height = 28;
    const smallContext = smallCanvas.getContext("2d");

    // Use pica to resize the canvas
    try {
      await picaInstance.resize(canvas, smallCanvas);

      let smallImageData = smallContext.getImageData(
        0,
        0,
        smallCanvas.width,
        smallCanvas.height
      );
      // console.log(smallImageData);
      let grayscaleValues = [];
      for (let i = 3; i < smallImageData.data.length; i += 4) {
        grayscaleValues.push(smallImageData.data[i]);
      }
      // console.log(grayscaleValues.length);
      return grayscaleValues;
    } catch (error) {
      message.info("Please disable shields for the prediction to work!");
      return null;
    }
  };

  async function predict() {
    setLoadingPredict(true);
    // await loadModel();
    const imageArray = await saveDrawing(); //imamgeArray is if length 784
    // console.log(imageArray);
    if (imageArray) {
      let digit = tf.tensor2d(imageArray, [1, 784], "float32");
      // console.log(digit);
      digit = digit.reshape([-1, 28, 28, 1]);
      const inference = model.predict(digit);
      const prediction = inference.argMax(1);
      const probablity = inference.max();
      setProb((probablity.dataSync()[0] * 100).toFixed(2));
      // probablity.print()
      setPred(prediction.dataSync()[0]);
      // console.log(prediction.dataSync()[0]);
      // setPred(prediction)
    } else {
      message.error("something went wrong!");
    }
    setLoadingPredict(false);
  }

  // now load the model
  useEffect(() => {
    class L2 {
      static className = "L2";
      constructor(config) {
        return tf.regularizers.l1l2(config);
      }
    }
    tf.serialization.registerClass(L2);
    loadModel();
    // console.log(digit);
  }, [loadModel]);

  const props = {
    onChange: null,
    loadTimeOffset: 0.0001,
    lazyRadius: 20,
    brushRadius: 25,
    brushColor: "#000",
    catenaryColor: "#fff",
    backgroundColor: "#fff",
    gridColor: "rgba(150,150,150,0.17)",
    hideGrid: true,
    canvasWidth: 400,
    canvasHeight: 400,
    disabled: false,
    imgSrc: "",
    saveData: null,
    immediateLoading: false,
    hideInterface: true,
    gridSizeX: 25,
    gridSizeY: 25,
    gridLineWidth: 0.5,
    hideGridX: false,
    hideGridY: false,
    enablePanAndZoom: false,
    mouseZoomFactor: 0.0,
    zoomExtents: { min: 0.33, max: 3 },
  };
  const steps = [
    {
      title: "Draw a Digit Here",
      description:
        "Draw a digit inside the canvas! preferably keep some distance from the border.",
      target: () => canvas.current,
    },
    {
      title: "Predict Digit",
      description: "Click to Predict the digit drawn on the Canvas!",
      target: () => predictButton.current,
    },
    {
      title: "Undo Button",
      description:
        "Undo the last Drawing on the Canvas Does not affect the current results",
      target: () => undoButton.current,
    },
    {
      title: "Clear the Drawing!",
      description: "Clear the canvas and the Results",
      target: () => clearButton.current,
    },

    {
      title: "Results",
      description:
        "The Predicted Digit will be shown here along the the Probability of the prediction.",
      target: () => results.current,
    },
  ];
  return (
    <div className="App">
      <Row gutter={[16, 16]}>
        <Col span={24}>
          <h1 style={{ textAlign: "center" }}>
            Handwritten Digit Image Prediction
          </h1>
          <h4 style={{ textAlign: "center" }}>Author : Saieed Shafi</h4>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginTop: 20, marginBottom: 25 }}>
        <Col xs={24} sm={24} md={8}>
          <Row gutter={[16, 16]}>
            <Col xs={24} sm={24} md={12} ref={results}>
              <Row justify="center">
                <Col span={24} style={{ textAlign: "center" }}>
                  <span style={{ color: "gray", fontSize: 20 }}>
                    Probability :
                  </span>
                  <span style={{ fontSize: 25, margin: 5 }}>
                    {prob}
                    {prob ? "%" : ""}
                  </span>
                </Col>
              </Row>
              <Row justify="center">
                <Col span={24} style={{ textAlign: "center" }}>
                  <h1 style={{ padding: 0, margin: 0, color: "gray" }}>
                    Prediction :
                  </h1>
                  <h1 style={{ fontSize: 200, padding: 0, margin: 0 }}>
                    {pred}
                  </h1>
                </Col>
              </Row>
            </Col>
          </Row>
        </Col>
        <Col xs={24} sm={24} md={16}>
          <Row justify="center">
            <Col span={24} style={{ textAlign: "center" }}>
              <Button type="link" onClick={() => setOpen(true)}>
                How to Use?
              </Button>
            </Col>
          </Row>
          <Row justify="centers">
            <Col span={24} ref={canvas}>
              <CanvasDraw
                {...props}
                ref={canvasRef}
                style={{ border: "2px solid black", margin: "auto" }}
              />
            </Col>
          </Row>
          <Row gutter={[20, 20]} justify="center" style={{ marginTop: 20 }}>
            <Col>
              <Button
                icon={<ClearOutlined />}
                onClick={() => {
                  canvasRef.current.eraseAll();
                  setPred();
                  setProb();
                }}
                type="primary"
                danger
                ref={clearButton}
              >
                Clear
              </Button>
            </Col>
            <Col>
              <Button
                icon={<UndoOutlined />}
                onClick={() => canvasRef.current.undo()}
                ref={undoButton}
              >
                Undo
              </Button>
            </Col>
            <Col>
              <Button
                type="primary"
                icon={<QuestionOutlined />}
                onClick={predict}
                loading={loadingPredict}
                ref={predictButton}
              >
                Predict Digit
              </Button>
            </Col>
          </Row>
        </Col>
      </Row>
      <Tour open={open} onClose={() => setOpen(false)} steps={steps} />
    </div>
  );
}

export default App;
