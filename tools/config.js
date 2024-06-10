import path from 'path';
import { fileURLToPath } from 'url';

const events = ['apply', 'wait', 'sms', 'email', 'hire'];
const entities = ['driver', 'us'];

// Calculate the length of the one-hot encoded sequence per event
const featureLength = entities.length * events.length; // 10
// maximum number of events in the sequences. we may need to convert to a variable length
// if this gets too complicated, but for now we can just assume the max is 5 and we will
// use that number to calculate padding.
const inputLength = 10; // 50

// paths
const __filePath = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filePath);
const dataPath = path.join(__dirname, '../data');
const modelPath = path.join(__dirname, '../model');

// data split
const trainingRatio = 0.7; // 70% training
const validationRatio = 0.15; // 15% validation, remaining 15% for testing

export {
  events,
  entities,
  inputLength,
  dataPath,
  modelPath,
  featureLength,
  trainingRatio,
  validationRatio,
};
