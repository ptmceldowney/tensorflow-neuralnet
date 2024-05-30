import path from 'path';
import { fileURLToPath } from 'url';

const events = ['apply', 'wait', 'sms', 'email', 'hire'];
const entities = ['driver', 'us'];

// Calculate the length of the one-hot encoded sequence per event
const eventEncodingLength = entities.length * events.length; // 10
// maximum number of events in the sequences. we may need to convert to a variable length
// if this gets too complicated, but for now we can just assume the max is 5 and we will
// use that number to calculate padding.
const maxEvents = events.length; // 5
const inputLength = maxEvents * eventEncodingLength; // 50

// paths
const __filePath = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filePath);
const trainingDataPath = path.join(__dirname, '../data/training.json');
const modelPath = path.join(__dirname, '../model');

export {
  events,
  entities,
  inputLength,
  trainingDataPath,
  modelPath,
  eventEncodingLength,
};
