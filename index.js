'use strict';
const { create, } = require('./lib/index');
const Promisie = require('promisie');
const AWS = require('aws-sdk');

/**
 * Creates machine learning evaluators
 * @param {Object|Object[]} configurations A single configuration object or an array of configuration objects
 * @return {Object|Function} A single evaluator or an object containing evalutors indexed by name
 */
var generate = function (options, cb) {
  try {
    let evaluations;
    let { segments, module_name, machinelearning, } = options;
    if (!segments) throw new Error('No segments to evaluate');
    if (!Array.isArray(segments)) evaluations = create(segments, module_name, machinelearning);
    else {
      evaluations = segments.reduce((result, configuration) => {
        result[ configuration.name ] = create(configuration, module_name, machinelearning);
        return result;
      }, {});
    }
    return (typeof cb === 'function')
      ? cb(null, evaluations)
      : Promisie.resolve(evaluations);
  } catch (e) {
    return (typeof cb === 'function') ? cb(e) : Promisie.reject(e);
  }
};

module.exports = generate;
