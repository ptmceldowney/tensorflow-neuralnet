import { dataPath, trainingRatio, validationRatio } from '../tools/config.js';
import { readFileSync, writeFileSync } from 'fs';

const trainingData = JSON.parse(
  readFileSync(`${dataPath}/events.json`, 'utf8')
);

// split data into Training, validation, and test sets
const trainSize = Math.floor(trainingData.length * trainingRatio);
const valSize = Math.floor(trainingData.length * validationRatio);

// shuffle data in case it's ordered in anyway, this would through off the model fitting
const shuffledData = trainingData.sort(() => 0.5 - Math.random());
const trainData = shuffledData.slice(0, trainSize);
const valData = shuffledData.slice(trainSize, trainSize + valSize);
const testData = shuffledData.slice(trainSize + valSize);

writeFileSync(`${dataPath}/train.json`, JSON.stringify(trainData, null, 2));
writeFileSync(`${dataPath}/validation.json`, JSON.stringify(valData, null, 2));
writeFileSync(`${dataPath}/test.json`, JSON.stringify(testData, null, 2));
