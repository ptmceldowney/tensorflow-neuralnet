import * as tf from '@tensorflow/tfjs-node';
import { readFileSync, writeFileSync } from 'fs';
import yargs from 'yargs/yargs';
import { hideBin } from 'yargs/helpers';
import {
  createModel,
  loadModel,
  loadTrainingData,
  oneHotEncode,
  padSequence,
} from './tools/index.js';
import { modelPath, trainingDataPath, inputLength } from './tools/config.js';

/**
 * Trains an existing model or creates a new model based on the cli argument
 * @param {Boolean} newModel
 */
async function trainModel(newModel) {
  try {
    const model = await (newModel ? createModel() : loadModel());
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

/**
 * Function to make a prediction based on a sequence of events
 * @param {string} sequence - The sequence of events
 */
async function makePrediction(sequence) {
  try {
    const model = await loadModel();
    const encodedSequence = padSequence(oneHotEncode(sequence), inputLength);
    const inputTensor = tf.tensor2d([encodedSequence], [1, inputLength]);
    const prediction = await model.predict(inputTensor);
    const predictedValue = prediction.dataSync()[0]; // Extract the prediction value

    // Define a threshold, this can be adjusted if needed just needed a better way to display the prediction
    const threshold = 0.5;
    const result = predictedValue > threshold ? 'Hired' : 'Not hired';
    console.log(`Predicted probability: ${predictedValue}`);
    console.log(`Prediction: ${result}`);
  } catch (e) {
    console.error('Error making prediction', e);
  }
}

// Use yargs to parse CLI arguments
// node . train
// node train add -s 'driver:apply|us:sms|driver:hire' -o 1
// node train predict -s 'driver:apply|us:sms|driver:hire'
yargs(hideBin(process.argv))
  .command(
    'train',
    'Train the model',
    {
      newModel: {
        description: 'Create a new model instead of loading an existing one',
        alias: 'new',
        type: 'boolean',
        deafult: false,
      },
    },
    argv => {
      trainModel(argv.newModel);
    }
  )
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
  .command(
    'predict',
    'Make a prediction based on a sequence of events',
    {
      sequence: {
        description: 'The sequence of events (driver:apply|us:sms|driver:hire)',
        alias: 's',
        type: 'string',
        demandOption: true,
      },
    },
    argv => {
      makePrediction(argv.sequence);
    }
  )
  .demandCommand(1, 'You need to specify a command (train or add)')
  .help().argv;
