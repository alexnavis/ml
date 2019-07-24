'use strict';
const Promisie = require('promisie');
const utility = require('../utility');
const { evaluate, } = utility;

var generateAI = function (configuration, module_name, machinelearning) {
  return function evaluator(state) {
    try {
      let result = evaluate(configuration, state, machinelearning) || {};
      if (result.err) {
        state.error = {
          code: '',
          message: result.err.message,
        };
        result = {
          'type': 'artificialintelligence',
          'name': module_name,
          segment: configuration.name,
          classification: '',
          prediction: {},
          output: {},
          output_variable: configuration.output_variable,
        };
      } else {
        result = {
          'type': 'artificialintelligence',
          'name': module_name,
          segment: configuration.name,
          prediction: result.prediction,
          output: result.output,
          classification: result.classification,
          output_variable: configuration.output_variable,
        };
      }
      return result;
    } catch (e) {
      return Promisie.reject(e);
    }
  }
}

module.exports = generateAI;