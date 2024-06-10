import { readFileSync } from 'fs';
import {
  dataPath,
  entities,
  events,
  featureLength,
  inputLength,
} from './config.js';
import * as tf from '@tensorflow/tfjs-node';
import {
  loadModel,
  oneHotEncode,
  padSequence,
  preprocessData,
} from './index.js';
function loadTestData() {
  const rawData = readFileSync(`${dataPath}/test.json`, 'utf8');
  const testData = JSON.parse(rawData);
  return testData;
}

async function evaluateModel() {
  const testData = loadTestData();
  const { inputTensor, outputTensor } = preprocessData(testData);

  const model = await loadModel();

  // make predictions
  const predictions = model.predict(inputTensor);

  // evaluate predictions
  const predictedClasses = predictions.argMax(-1);
  const trueClasses = outputTensor.argMax(-1);
  const accuracy = predictedClasses.equal(trueClasses).mean().dataSync()[0];

  console.log(`Test accuracy: ${(accuracy * 100).toFixed(2)}%`);
}

/**
 * Function to make a prediction based on a sequence of events
 * @param {string} sequence - The sequence of events
 */
async function makePrediction(sequence) {
  try {
    const encodedSequence = padSequence(oneHotEncode(sequence), inputLength);
    const inputTensor = tf.tensor3d(
      [encodedSequence],
      [1, inputLength, featureLength]
    );

    const model = await loadModel();
    const prediction = model.predict(inputTensor);
    const predictedIndex = prediction.argMax(-1).dataSync()[0];

    const eventIndex = predictedIndex % events.length;
    const entityIndex = Math.floor(predictedIndex / events.length);
    const nextStep = `${entities[entityIndex]}:${events[eventIndex]}`;

    console.log(`Predicted next step: ${nextStep}`);
  } catch (e) {
    console.error('Error making prediction', e);
  }
}

export { evaluateModel, makePrediction };
