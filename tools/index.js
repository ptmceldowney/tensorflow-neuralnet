import * as tf from '@tensorflow/tfjs-node';
import {
  modelPath,
  inputLength,
  entities,
  events,
  eventEncodingLength,
} from './config.js';

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
 * Attempts to load a saved model, if not found it will create a new one
 * @returns {tf.Sequential | tf.LayersModel}
 */
async function loadModel() {
  let model;

  // Check if a saved model exists and load it
  try {
    model = await tf.loadLayersModel(`file://${modelPath}/model.json`);

    const optimizer = tf.train.adam(0.001);
    model.compile({
      optimizer: optimizer,
      loss: 'categoricalCrossentropy',
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
      units: 128,
      activation: 'relu',
      inputShape: [inputLength],
    })
  );
  model.add(tf.layers.dropout({ rate: 0.4 }));
  model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.4 }));
  model.add(
    tf.layers.dense({ units: eventEncodingLength, activation: 'softmax' })
  );

  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });
  return model;
}

export { loadModel, createModel, oneHotEncode, padSequence };
