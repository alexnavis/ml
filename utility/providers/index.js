'use strict';

const aws = require('./aws');
const decision_tree = require('./decision_tree');
const neural_network = require('./neural_network');
const random_forest = require('./random_forest');
const sagemaker_ll = require('./sagemaker_ll');
const sagemaker_xgb = require('./sagemaker_xgb');

module.exports = {
  aws,
  decision_tree,
  neural_network,
  random_forest,
  sagemaker_ll,
  sagemaker_xgb,
};