'use strict';
const chai = require('chai');
const expect = chai.expect;
const Promisie = require('promisie');
const MOCKS = require('../mocks');
const path = require('path');
const GENERATE = require(path.join(__dirname, '../../index'));
chai.use(require('chai-spies'));

describe('integration of output evaluators', function () {
  describe('generation of a single evaluator', function () {
    it('should be able to generate a single evaluator with a callback', done => {
      GENERATE({ segments: MOCKS.DEFAULT }, (err, evaluator) => {
        try {
          if (err) throw err;
          else {
            expect(evaluator).to.be.a('function');
            done();
          }
        } catch (e) {
          done(e);
        }
      });
    });
    it('should be able to generate a single evaluator and return a Promise', done => {
      GENERATE({ segments: MOCKS.DEFAULT })
        .try(evaluator => {
          expect(evaluator).to.be.a('function');
          done();
        })
        .catch(done);
    });
  });
  describe('generation of an array of evaluators', function () {
    it('should be able to generate a single evaluator with a callback', done => {
      GENERATE({ segments: [ MOCKS.DEFAULT, MOCKS.BASIC ] }, (err, evaluators) => {
        try {
          if (err) throw err;
          else {
            expect(evaluators).to.have.property('employee_hiring_model');
            expect(evaluators).to.have.property('employee_hiring_model_2');
            done();
          }
        } catch (e) {
          done(e);
        }
      });
    });
    it('should be able to generate a single evaluator and return a Promise', done => {
      GENERATE({ segments: [ MOCKS.DEFAULT, MOCKS.BASIC ] })
        .try(evaluators => {
          expect(evaluators).to.have.property('employee_hiring_model');
          expect(evaluators).to.have.property('employee_hiring_model_2');
          done();
        })
        .catch(done);
    });
  });
  describe('handling errors in generation process', function () {
    it('should handle an error when a callback is passed', done => {
      GENERATE({ segments: null }, err => {
        try {
          expect(err).to.be.instanceof(Error);
          done();
        } catch (e) {
          done(e);
        }
      });
    });
    it('should handle an error when no callback is passed', done => {
      GENERATE({ segments: null })
        .then(() => {
          done(new Error('should not execute'));
        }, e => {
          expect(e).to.be.instanceof(Error);
          done();
        });
    });
  });
});