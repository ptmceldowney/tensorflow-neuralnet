import * as tf from '@tensorflow/tfjs-node';
import { readFileSync, writeFileSync } from 'fs';
import yargs from 'yargs/yargs';
import { hideBin } from 'yargs/helpers';
import {
  createModel,
  loadModel,
  oneHotEncode,
  padSequence,
} from './tools/index.js';
import {
  modelPath,
  trainingDataPath,
  inputLength,
  eventEncodingLength,
  events,
  entities,
} from './tools/config.js';

/**
 * Trains an existing model or creates a new model based on the cli argument
 * @param {Boolean} newModel
 */
async function trainModel(newModel) {
  try {
    const model = await (newModel ? createModel() : loadModel());
    const trainingData = JSON.parse(readFileSync(trainingDataPath, 'utf8'));

    // Encode and pad the training data
    const inputs = trainingData.map(data =>
      padSequence(oneHotEncode(data.input), inputLength)
    );
    const outputs = trainingData.map(data => oneHotEncode(data.output));

    const xs = tf.tensor2d(inputs, [inputs.length, inputLength]);
    const ys = tf.tensor2d(outputs, [outputs.length, eventEncodingLength]);

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
    const outputs = currentData.map(data => [data.output]);

    const xs = tf.tensor2d(inputs, [inputs.length, inputLength]);
    const ys = tf.tensor2d(outputs, [outputs.length, eventEncodingLength]);

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
    const encodedSequence = padSequence(oneHotEncode(sequence), inputLength);
    const inputTensor = tf.tensor2d([encodedSequence], [1, inputLength]);

    const model = await loadModel();
    const prediction = model.predict(inputTensor);
    const predictedIndex = prediction.argMax(-1).dataSync()[0];

    const eventindex = predictedIndex % events.length;
    const entityIndex = Math.floor(predictedIndex / events.length);
    const nextStep = `${entities[entityIndex]}:${events[eventindex]}`;

    console.log(`Predicted next step: ${nextStep}`);
  } catch (e) {
    console.error('Error making prediction', e);
  }
}

// Use yargs to parse CLI arguments
// 1. Train new model: node main.js train -new
// 2  Train existing model (creates a new one if none exists): node main.js train
// 3. Add new training data: node main.js train add -s 'driver:apply|us:sms|driver:hire' -o 1
// 4. Make prediction: node main.js predict -s 'driver:apply|us:sms|driver:hire'
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
