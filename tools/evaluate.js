import { readFileSync } from 'fs';
import { dataPath } from './config.js';
import { loadModel, preprocessData } from './index.js';
function loadTestData() {
  const rawData = readFileSync(`${dataPath}/test.json`, 'utf8');
  const testData = JSON.parse(rawData);
  return testData;
}

async function evaluateModel() {
  const testData = loadTestData();
  const { inputTensor, outputTensor } = preprocessData(testData);

  const model = await loadModel();

  // make predictions
  const predictions = model.predict(inputTensor);

  // evaluate predictions
  const predictedClasses = predictions.argMax(-1);
  const trueClasses = outputTensor.argMax(-1);
  const accuracy = predictedClasses.equal(trueClasses).mean().dataSync()[0];

  console.log(`Test accuracy: ${(accuracy * 100).toFixed(2)}%`);
}

export { evaluateModel };
