'use strict';
const chai = require('chai');
const expect = chai.expect;
const Promisie = require('promisie');
const MOCKS = require('../mocks');
const path = require('path');
const CREATE_EVALUATOR = require(path.join(__dirname, '../../lib')).create;

chai.use(require('chai-spies'));

describe('artificial intelligence module', function () {
  function binary_predict(params, cb) {
    let prediction = require('../mocks')[ 'BINARY' ];
    if (typeof cb === 'function') cb(null, prediction);
  }
  function linear_predict(params, cb) {
    let prediction = require('../mocks')[ 'LINEAR' ];
    if (typeof cb === 'function') cb(null, prediction);
  }
  function categorical_predict(params, cb) {
    let prediction = require('../mocks')[ 'CATEGORICAL' ];
    if (typeof cb === 'function') cb(null, prediction);
  }
  function binary_predict_error(params, cb) {
    let prediction = require('../mocks')[ 'BINARY_ERROR' ];
    if (typeof cb === 'function') cb(null, prediction);
  }
  function linear_predict_error(params, cb) {
    let prediction = require('../mocks')[ 'LINEAR_ERROR' ];
    if (typeof cb === 'function') cb(null, prediction);
  }
  function categorical_predict_error(params, cb) {
    let prediction = require('../mocks')[ 'CATEGORICAL_ERROR' ];
    if (typeof cb === 'function') cb(null, prediction);
  }
  describe('basic assumptions', function () {
    it('should have a create method that is a function', () => {
      expect(CREATE_EVALUATOR).to.be.a('function');
    });
    it('should accept a segment as an arguments and generate an evaluator', () => {
      let evaluator = CREATE_EVALUATOR(MOCKS.DEFAULT, 'employee_hiring_model', { predict: binary_predict, });
      expect(evaluator).to.be.a('function');
    });
  });
});