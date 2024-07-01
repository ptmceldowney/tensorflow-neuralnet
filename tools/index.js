import * as tf from '@tensorflow/tfjs-node';
import {
  modelPath,
  inputLength,
  entities,
  events,
  featureLength,
} from './config.js';

/**
 * Function to one-hot encode a sequence of events
 * @param {string} sequence - The sequence of events
 * @returns {Array<number>} - The one-hot encoded representation of the sequence
 */
function oneHotEncode(sequence) {
  // Create one-hot encoded sequence
  const encodedSequence = [];
  sequence.split('|').forEach(step => {
    const [entity, event] = step.split(':');
    const entityIndex = entities.indexOf(entity);
    const eventIndex = events.indexOf(event);

    const oneHotArray = Array(featureLength).fill(0);
    oneHotArray[entityIndex * events.length + eventIndex] = 1;

    encodedSequence.push(oneHotArray);
  });

  return encodedSequence;
}

/**
 * Function to pad encoded sequences to a fixed length
 * @param {Array<number>} sequence - The one-hot encoded sequence
 * @param {number} maxLength - The length to pad the sequence to
 * @returns {Array<number>} - The padded sequence
 */
function padSequence(sequence, inputLength) {
  while (sequence.length < inputLength) {
    sequence.push(Array(featureLength).fill(0));
  }

  return sequence.slice(0, inputLength);
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
      optimizer: tf.train.adam(),
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
    tf.layers.lstm({
      units: 64,
      returnSequences: true,
      inputShape: [inputLength, featureLength],
    })
  );
  model.add(tf.layers.dropout({ rate: 0.3 }));

  model.add(
    tf.layers.lstm({
      units: 64,
    })
  );
  model.add(tf.layers.dropout({ rate: 0.3 }));

  model.add(
    tf.layers.dense({
      units: featureLength,
      activation: 'softmax',
    })
  );

  model.compile({
    optimizer: tf.train.adam(),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });
  return model;
}

/**
 * Converts data into input and output tensors
 * @param {*} data
 * @returns
 */
function preprocessData(data) {
  const inputs = data.map(d =>
    padSequence(
      oneHotEncode(d.input, entities, events),
      inputLength,
      featureLength
    )
  );
  const outputs = data.map(d => oneHotEncode(d.output, entities, events)[0]); // Output is a single step

  const inputTensor = tf.tensor3d(inputs, [
    inputs.length,
    inputLength,
    featureLength,
  ]);
  const outputTensor = tf.tensor2d(outputs, [outputs.length, featureLength]);

  return { inputTensor, outputTensor };
}

export { loadModel, createModel, oneHotEncode, padSequence, preprocessData };
