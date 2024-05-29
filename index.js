import * as tf from '@tensorflow/tfjs-node';
import path from 'path';
import { fileURLToPath } from 'url';
import { readFileSync, writeFileSync } from 'fs';
import yargs from 'yargs/yargs';
import { hideBin } from 'yargs/helpers';

const __filePath = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filePath);
const trainingDataPath = path.join(__dirname, 'data/training.json');
const modelPath = path.join(__dirname, 'model');

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
  model.add(tf.layers.dense({ units: 10, activation: 'relu' }));
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
 * Train the model
 */
async function trainModel() {
  try {
    const model = await loadModel();
    const { xs, ys } = loadTrainingData();
    await model.fit(xs, ys, {
      epochs: 100,
      callbacks: {
        onEpochEnd: (epoch, log) =>
          console.log(`Epoch ${epoch}: loss = ${log.loss}`),
      },
    });

    // save the model
    await model.save(`file://${modelPath}`);
    console.log('\nModel saved to ', modelPath);

    // Example prediciton
    const newSequence = 'driver:apply|us:sms|driver:hire';
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
    console.error('error training model', error);
  }
}

/**
 * Function to add new training data and retrain the model
 * @param {string} sequence - The sequence of events
 * @param {number} output - The actual outcome (0 or 1)
 */
async function addTrainingData(sequence, output) {
  try {
    // Load the current training data
    const currentData = JSON.parse(readFileSync(trainingDataPath, 'utf8'));

    currentData.push({ input: sequence, output: output });

    // Save updated training data
    writeFileSync(
      trainingDataPath,
      JSON.stringify(currentData, null, 2),
      'utf8'
    );

    // Encode and pad the updated training data
    const inputs = currentData.map(data =>
      padSequence(oneHotEncode(data.input), inputLength)
    );
    const labels = currentData.map(data => [data.output]);

    const xs = tf.tensor2d(inputs, [inputs.length, inputLength]);
    const ys = tf.tensor2d(labels, [labels.length, 1]);

    let model = await loadModel();
    await model.fit(xs, ys, {
      epochs: 100,
      callbacks: {
        onEpochEnd: (epoch, log) =>
          console.log(`Epoch ${epoch}: loss = ${log.loss}`),
      },
    });

    // Save the retrained model
    await model.save(`file://${modelPath}`);
    console.log(`Model retrained and saved to ${modelPath}`);
  } catch (error) {
    console.error('Error adding new training data', error);
  }
}

// Use yargs to parse CLI arguments
// node . train
// node . add -s 'driver:apply|us:sms|driver:hire' -o 1
yargs(hideBin(process.argv))
  .command('train', 'Train the model from scratch', {}, () => {
    trainModel();
  })
  .command(
    'add',
    'Add new training data and retrain the model',
    {
      sequence: {
        description: 'The sequence of events (driver:apply|us:sms|driver:hire)',
        alias: 's',
        type: 'string',
        demandOption: true,
      },
      output: {
        description: 'The actual outcome (0 or 1)',
        alias: 'o',
        type: 'number',
        demandOption: true,
      },
    },
    argv => {
      addTrainingData(argv.sequence, argv.output);
    }
  )
  .demandCommand(1, 'You need to specify a command (train or add)')
  .help().argv;
