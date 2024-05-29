import * as tf from '@tensorflow/tfjs-node';
import {
  modelPath,
  inputLength,
  trainingDataPath,
  entities,
  events,
} from './config.js';
import { readFileSync } from 'fs';

/**
 * Function to one-hot encode a sequence of events
 * @param {string} sequence - The sequence of events
 * @returns {Array<number>} - The one-hot encoded representation of the sequence
 */
function oneHotEncode(sequence) {
  const eventMap = {};
  let index = 0;

  // Create an index map for events
  entities.forEach(entity => {
    events.forEach(event => {
      eventMap[`${entity}:${event}`] = index++;
    });
  });

  // Create one-hot encoded sequence
  const encodedSequence = [];
  sequence.split('|').forEach(event => {
    const encoding = new Array(index).fill(0);
    if (eventMap[event] !== undefined) {
      encoding[eventMap[event]] = 1;
    }
    encodedSequence.push(encoding);
  });

  return encodedSequence.flat();
}

/**
 * Function to pad encoded sequences to a fixed length
 * @param {Array<number>} sequence - The one-hot encoded sequence
 * @param {number} maxLength - The length to pad the sequence to
 * @returns {Array<number>} - The padded sequence
 */
function padSequence(sequence, maxLength) {
  if (sequence.length >= maxLength) {
    return sequence.slice(0, maxLength);
  } else {
    return sequence.concat(new Array(maxLength - sequence.length).fill(0));
  }
}

/**
 * Loads training data and encodes it into inputs and labels
 * @returns {{xs: tf.Tensor2D, ys: tf.Tensor2D}}
 */
function loadTrainingData() {
  const trainingData = JSON.parse(readFileSync(trainingDataPath, 'utf8'));
  // need to pad the sequence for varying encoding lengths
  const inputs = trainingData.map(data =>
    padSequence(oneHotEncode(data.input), inputLength)
  );
  const labels = trainingData.map(data => data.output);
  const xs = tf.tensor2d(inputs, [inputs.length, inputLength]);
  const ys = tf.tensor2d(labels, [labels.length, 1]);
  return { xs, ys };
}

/**
 * Attempts to load a saved model, if not found it will create a new one
 * @returns {tf.Sequential | tf.LayersModel}
 */
async function loadModel() {
  let model;

  // Check if a saved model exists and load it
  try {
    model = await tf.loadLayersModel(`file://${modelPath}/model.json`);
    model.compile({
      optimizer: 'adam',
      loss: 'binaryCrossentropy',
      metrics: ['accuracy'],
    });
    console.log('Model loaded from file');
  } catch (error) {
    console.log('No saved model found, creating a new one');
    model = createModel();
  }

  return model;
}

/**
 * Create and compile the model
 * @returns {tf.Sequential} - The compiled model
 */
function createModel() {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      units: 10,
      activation: 'relu',
      inputShape: [inputLength],
    })
  );
  model.add(tf.layers.dense({ units: 10, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

  model.compile({
    optimizer: 'adam',
    loss: 'binaryCrossentropy',
    metrics: ['accuracy'],
  });
  return model;
}

export { loadTrainingData, loadModel, createModel, oneHotEncode, padSequence };
