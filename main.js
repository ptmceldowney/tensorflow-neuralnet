import * as tf from '@tensorflow/tfjs-node';
import { readFileSync, writeFileSync } from 'fs';
import yargs from 'yargs/yargs';
import { hideBin } from 'yargs/helpers';
import {
  createModel,
  loadModel,
  oneHotEncode,
  padSequence,
  preprocessData,
} from './tools/index.js';
import {
  modelPath,
  dataPath,
  inputLength,
  featureLength,
  events,
  entities,
} from './tools/config.js';
import { evaluateModel } from './tools/evaluate.js';

/**
 * Trains an existing model or creates a new model based on the cli argument
 * @param {Boolean} newModel
 */
async function trainModel(newModel) {
  try {
    const model = await (newModel ? createModel() : loadModel());

    const trainingData = JSON.parse(
      readFileSync(`${dataPath}/train.json`, 'utf8')
    );
    const valData = JSON.parse(
      readFileSync(`${dataPath}/validation.json`, 'utf8')
    );

    // Encode and pad the training data
    const { inputTensor: trainXs, outputTensor: trainYs } =
      preprocessData(trainingData);
    const { inputTensor: valXs, outputTensor: valYs } = preprocessData(valData);

    await model.fit(trainXs, trainYs, {
      epochs: 100,
      validationData: [valXs, valYs],
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
    const currentData = JSON.parse(
      readFileSync(`${dataPath}/train.json`, 'utf8')
    );

    currentData.push({ input: sequence, output: output });

    // Save updated training data
    writeFileSync(
      `${dataPath}/train.json`,
      JSON.stringify(currentData, null, 2),
      'utf8'
    );

    trainModel();
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
        demandOption: false,
      },
    },
    argv => {
      if (argv.sequence) {
        makePrediction(argv.sequence);
      } else {
        evaluateModel();
      }
    }
  )
  .demandCommand(1, 'You need to specify a command (train or add)')
  .help().argv;
