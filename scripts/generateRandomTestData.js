import { writeFileSync } from 'fs';
import {
  entities,
  events,
  dataPath,
  validationRatio,
  trainingRatio,
} from '../tools/config.js';

/**
 * Function to generate a random event sequence
 * @param {*} maxLength
 * @returns
 */
function generateRandomRequence(maxLength) {
  const sequenceLength = Math.floor(Math.random() * maxLength) + 1;
  const sequence = [];

  for (let i = 0; i < sequenceLength; i++) {
    const randomEntity = entities[Math.floor(Math.random() * entities.length)];
    let randomEvent;

    if (randomEntity === 'driver') {
      // Ensure that the first event is not driver:hire and driver:wait isn't an event anytime
      randomEvent = events.filter(
        event => event !== 'wait' || (i === 0 && event !== 'hire')
      )[Math.floor(Math.random() * (events.length - 1))];
    } else {
      // Ensure that only drivers have 'apply' and 'hire' events
      randomEvent = events.filter(
        event => event !== 'apply' && event !== 'hire'
      )[Math.floor(Math.random() * (events.length - 2))];
    }

    sequence.push(`${randomEntity}:${randomEvent}`);

    // if event is driver:hire we want to break out of the loop because we are done
    if (sequence[sequence.length - 1] === 'driver:hire') break;
  }

  return sequence.join('|');
}

/**
 * Function to generate random training data
 * @param {*} numSamples
 * @param {*} maxLength
 * @returns
 */
function generateRandomData(numSamples, maxLength) {
  const data = [];
  for (let i = 0; i < numSamples; i++) {
    const inputSequence = generateRandomRequence(maxLength);
    const outputSequence = generateRandomRequence(1);
    data.push({ input: inputSequence, output: outputSequence });
  }

  return data;
}

/**
 * Split data into training, validation, and test sets
 * @param {*} data
 * @returns
 */
function splitData(data) {
  const trainSize = Math.floor(data.length * trainingRatio);
  const valSize = Math.floor(data.length * validationRatio);

  const trainingData = data.slice(0, trainSize);
  const validationData = data.slice(trainSize, trainSize + valSize);
  const testData = data.slice(trainSize + valSize);

  return { trainingData, validationData, testData };
}

/**
 * Generate and save random training and validation data
 * @param {*} trainingSamples
 * @param {*} validationSamples
 * @param {*} maxLength
 */
function generateAndSaveData(totalSamples, maxLength) {
  const data = generateRandomData(totalSamples, maxLength);
  const { trainingData, validationData, testData } = splitData(
    data,
    trainingRatio,
    validationRatio
  );

  writeFileSync(
    `${dataPath}/train.json`,
    JSON.stringify(trainingData, null, 2)
  );
  writeFileSync(
    `${dataPath}/validation.json`,
    JSON.stringify(validationData, null, 2)
  );
  writeFileSync(`${dataPath}/test.json`, JSON.stringify(testData, null, 2));

  console.log(
    `Generated ${trainingData.length} training samples, ${validationData.length} validation samples, and ${testData.length} test samples.`
  );
}

const totalSamples = 1400;
const maxLength = 10; // Maximum length of the input sequences

generateAndSaveData(totalSamples, maxLength);
