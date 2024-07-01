import { readFileSync, writeFileSync } from 'fs';
import yargs from 'yargs/yargs';
import { hideBin } from 'yargs/helpers';
import { createModel, loadModel, preprocessData } from './tools/index.js';
import { modelPath, dataPath } from './tools/config.js';
import { evaluateModel, makePrediction } from './tools/evaluate.js';

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
      epochs: 50,
      batchSize: 32,
      validationData: [valXs, valYs],
      verbose: 1,
    });

    // save the model
    await model.save(`file://${modelPath}`);
    console.log('\nModel saved to ', modelPath);
  } catch (error) {
    console.error('error training model', error);
  }
}
// Use yargs to parse CLI arguments
// 1. Train new model: node main.js train -new
// 2. Train existing model (creates a new one if none exists): node main.js train
// 3. Evalulate on test data: node main.js predicr
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
