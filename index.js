import * as tf from '@tensorflow/tfjs-node';
import path from 'path';
import { fileURLToPath } from 'url';
import { readFileSync } from 'fs';

const __filePath = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filePath);

const events = ['apply', 'wait', 'sms', 'email', 'hire'];
const entities = ['driver', 'us'];

// Calculate the length of the one-hot encoded sequence per event
const eventEncodingLength = entities.length * events.length; // 10
// maximum number of events in the sequences. we may need to convert to a variable length
// if this gets too complicated, but for now we can just assume the max is 5 and we will
// use that number to calculate padding.
const maxEvents = events.length; // 5
const inputLength = maxEvents * eventEncodingLength; // 50

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
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
  model.compile({
    optimizer: 'adam',
    loss: 'binaryCrossentropy',
    metrics: ['accuracy'],
  });
  return model;
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

/*
 * TODO: import data from a file here
 * Loads training data and encodes it into inputs and labels
 * @returns {Object}
 */
function loadTrainingData() {
  const trainingDataPath = path.join(__dirname, 'data/training.json');
  const trainingData = JSON.parse(readFileSync(trainingDataPath, 'utf8'));
  console.log(trainingData);
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
 * Train the model
 */
async function trainModel() {
  try {
    const { xs, ys } = loadTrainingData();
    const model = createModel();
    await model.fit(xs, ys, {
      epochs: 100,
      callbacks: {
        onEpochEnd: (epoch, log) =>
          console.log(`Epoch ${epoch}: loss = ${log.loss}`),
      },
    });

    model.summary();

    // save the model

    const modelPath = path.join(__dirname, 'model');
    await model.save(`file://${modelPath}`);
    console.log('\nModel saved to ', modelPath);

    // Example prediciton
    const newSequence = 'driver:apply|us:wait|us:sms';
    const encodedNewSequence = padSequence(
      oneHotEncode(newSequence),
      inputLength
    );
    const inputTensor = tf.tensor2d([encodedNewSequence], [1, inputLength]);

    const prediction = await model.predict(inputTensor);
    const predictedValue = prediction.dataSync()[0]; // Extract the prediction value

    // Define a threshold, this can be adjusted if needed just needed a better way to display the prediction
    const threshold = 0.5;
    const result = predictedValue > threshold ? 'hire' : 'not hire';
    console.log(`Predicted probability: ${predictedValue}`);
    console.log(`Prediction: ${result}`);
  } catch (error) {
    console.error(error);
  }
}

trainModel();
