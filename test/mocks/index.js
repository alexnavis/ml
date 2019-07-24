
'use strict';
const BASIC = require('./basic.json');
const DEFAULT = require('./default.json');
const BINARY = require('./binary_prediction.json');
const BINARY_ERROR = require('./binary_prediction_error.json');
const LINEAR = require('./linear_prediction.json');
const LINEAR_ERROR = require('./linear_prediction_error.json');
const CATEGORICAL = require('./categorical_prediction.json');
const CATEGORICAL_ERROR = require('./categorical_prediction_error.json');

module.exports = {
  BASIC,
  BINARY,
  BINARY_ERROR,
  DEFAULT,
  LINEAR,
  LINEAR_ERROR,
  CATEGORICAL,
  CATEGORICAL_ERROR,
};