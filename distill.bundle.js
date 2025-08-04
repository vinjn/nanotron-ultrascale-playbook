/******/ (() => { // webpackBootstrap
/******/ 	var __webpack_modules__ = ({

/***/ 792:
/***/ ((module, exports, __webpack_require__) => {

var __WEBPACK_AMD_DEFINE_FACTORY__, __WEBPACK_AMD_DEFINE_RESULT__;function _wrapNativeSuper(t) { var r = "function" == typeof Map ? new Map() : void 0; return _wrapNativeSuper = function _wrapNativeSuper(t) { if (null === t || !_isNativeFunction(t)) return t; if ("function" != typeof t) throw new TypeError("Super expression must either be null or a function"); if (void 0 !== r) { if (r.has(t)) return r.get(t); r.set(t, Wrapper); } function Wrapper() { return _construct(t, arguments, _getPrototypeOf(this).constructor); } return Wrapper.prototype = Object.create(t.prototype, { constructor: { value: Wrapper, enumerable: !1, writable: !0, configurable: !0 } }), _setPrototypeOf(Wrapper, t); }, _wrapNativeSuper(t); }
function _construct(t, e, r) { if (_isNativeReflectConstruct()) return Reflect.construct.apply(null, arguments); var o = [null]; o.push.apply(o, e); var p = new (t.bind.apply(t, o))(); return r && _setPrototypeOf(p, r.prototype), p; }
function _isNativeFunction(t) { try { return -1 !== Function.toString.call(t).indexOf("[native code]"); } catch (n) { return "function" == typeof t; } }
function _toConsumableArray(r) { return _arrayWithoutHoles(r) || _iterableToArray(r) || _unsupportedIterableToArray(r) || _nonIterableSpread(); }
function _nonIterableSpread() { throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."); }
function _iterableToArray(r) { if ("undefined" != typeof Symbol && null != r[Symbol.iterator] || null != r["@@iterator"]) return Array.from(r); }
function _arrayWithoutHoles(r) { if (Array.isArray(r)) return _arrayLikeToArray(r); }
function _createForOfIteratorHelper(r, e) { var t = "undefined" != typeof Symbol && r[Symbol.iterator] || r["@@iterator"]; if (!t) { if (Array.isArray(r) || (t = _unsupportedIterableToArray(r)) || e && r && "number" == typeof r.length) { t && (r = t); var _n = 0, F = function F() {}; return { s: F, n: function n() { return _n >= r.length ? { done: !0 } : { done: !1, value: r[_n++] }; }, e: function e(r) { throw r; }, f: F }; } throw new TypeError("Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."); } var o, a = !0, u = !1; return { s: function s() { t = t.call(r); }, n: function n() { var r = t.next(); return a = r.done, r; }, e: function e(r) { u = !0, o = r; }, f: function f() { try { a || null == t["return"] || t["return"](); } finally { if (u) throw o; } } }; }
function _callSuper(t, o, e) { return o = _getPrototypeOf(o), _possibleConstructorReturn(t, _isNativeReflectConstruct() ? Reflect.construct(o, e || [], _getPrototypeOf(t).constructor) : o.apply(t, e)); }
function _possibleConstructorReturn(t, e) { if (e && ("object" == _typeof(e) || "function" == typeof e)) return e; if (void 0 !== e) throw new TypeError("Derived constructors may only return object or undefined"); return _assertThisInitialized(t); }
function _assertThisInitialized(e) { if (void 0 === e) throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); return e; }
function _isNativeReflectConstruct() { try { var t = !Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function () {})); } catch (t) {} return (_isNativeReflectConstruct = function _isNativeReflectConstruct() { return !!t; })(); }
function _superPropGet(t, o, e, r) { var p = _get(_getPrototypeOf(1 & r ? t.prototype : t), o, e); return 2 & r && "function" == typeof p ? function (t) { return p.apply(e, t); } : p; }
function _get() { return _get = "undefined" != typeof Reflect && Reflect.get ? Reflect.get.bind() : function (e, t, r) { var p = _superPropBase(e, t); if (p) { var n = Object.getOwnPropertyDescriptor(p, t); return n.get ? n.get.call(arguments.length < 3 ? e : r) : n.value; } }, _get.apply(null, arguments); }
function _superPropBase(t, o) { for (; !{}.hasOwnProperty.call(t, o) && null !== (t = _getPrototypeOf(t));); return t; }
function _getPrototypeOf(t) { return _getPrototypeOf = Object.setPrototypeOf ? Object.getPrototypeOf.bind() : function (t) { return t.__proto__ || Object.getPrototypeOf(t); }, _getPrototypeOf(t); }
function _inherits(t, e) { if ("function" != typeof e && null !== e) throw new TypeError("Super expression must either be null or a function"); t.prototype = Object.create(e && e.prototype, { constructor: { value: t, writable: !0, configurable: !0 } }), Object.defineProperty(t, "prototype", { writable: !1 }), e && _setPrototypeOf(t, e); }
function _setPrototypeOf(t, e) { return _setPrototypeOf = Object.setPrototypeOf ? Object.setPrototypeOf.bind() : function (t, e) { return t.__proto__ = e, t; }, _setPrototypeOf(t, e); }
function _typeof(o) { "@babel/helpers - typeof"; return _typeof = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function (o) { return typeof o; } : function (o) { return o && "function" == typeof Symbol && o.constructor === Symbol && o !== Symbol.prototype ? "symbol" : typeof o; }, _typeof(o); }
function _classCallCheck(a, n) { if (!(a instanceof n)) throw new TypeError("Cannot call a class as a function"); }
function _defineProperties(e, r) { for (var t = 0; t < r.length; t++) { var o = r[t]; o.enumerable = o.enumerable || !1, o.configurable = !0, "value" in o && (o.writable = !0), Object.defineProperty(e, _toPropertyKey(o.key), o); } }
function _createClass(e, r, t) { return r && _defineProperties(e.prototype, r), t && _defineProperties(e, t), Object.defineProperty(e, "prototype", { writable: !1 }), e; }
function _defineProperty(e, r, t) { return (r = _toPropertyKey(r)) in e ? Object.defineProperty(e, r, { value: t, enumerable: !0, configurable: !0, writable: !0 }) : e[r] = t, e; }
function _toPropertyKey(t) { var i = _toPrimitive(t, "string"); return "symbol" == _typeof(i) ? i : i + ""; }
function _toPrimitive(t, r) { if ("object" != _typeof(t) || !t) return t; var e = t[Symbol.toPrimitive]; if (void 0 !== e) { var i = e.call(t, r || "default"); if ("object" != _typeof(i)) return i; throw new TypeError("@@toPrimitive must return a primitive value."); } return ("string" === r ? String : Number)(t); }
function _slicedToArray(r, e) { return _arrayWithHoles(r) || _iterableToArrayLimit(r, e) || _unsupportedIterableToArray(r, e) || _nonIterableRest(); }
function _nonIterableRest() { throw new TypeError("Invalid attempt to destructure non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."); }
function _unsupportedIterableToArray(r, a) { if (r) { if ("string" == typeof r) return _arrayLikeToArray(r, a); var t = {}.toString.call(r).slice(8, -1); return "Object" === t && r.constructor && (t = r.constructor.name), "Map" === t || "Set" === t ? Array.from(r) : "Arguments" === t || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(t) ? _arrayLikeToArray(r, a) : void 0; } }
function _arrayLikeToArray(r, a) { (null == a || a > r.length) && (a = r.length); for (var e = 0, n = Array(a); e < a; e++) n[e] = r[e]; return n; }
function _iterableToArrayLimit(r, l) { var t = null == r ? null : "undefined" != typeof Symbol && r[Symbol.iterator] || r["@@iterator"]; if (null != t) { var e, n, i, u, a = [], f = !0, o = !1; try { if (i = (t = t.call(r)).next, 0 === l) { if (Object(t) !== t) return; f = !1; } else for (; !(f = (e = i.call(t)).done) && (a.push(e.value), a.length !== l); f = !0); } catch (r) { o = !0, n = r; } finally { try { if (!f && null != t["return"] && (u = t["return"](), Object(u) !== u)) return; } finally { if (o) throw n; } } return a; } }
function _arrayWithHoles(r) { if (Array.isArray(r)) return r; }
(function (factory) {
   true ? !(__WEBPACK_AMD_DEFINE_FACTORY__ = (factory),
		__WEBPACK_AMD_DEFINE_RESULT__ = (typeof __WEBPACK_AMD_DEFINE_FACTORY__ === 'function' ?
		(__WEBPACK_AMD_DEFINE_FACTORY__.call(exports, __webpack_require__, exports, module)) :
		__WEBPACK_AMD_DEFINE_FACTORY__),
		__WEBPACK_AMD_DEFINE_RESULT__ !== undefined && (module.exports = __WEBPACK_AMD_DEFINE_RESULT__)) : 0;
})(function () {
  'use strict';

  // Copyright 2018 The Distill Template Authors
  //
  // Licensed under the Apache License, Version 2.0 (the "License");
  // you may not use this file except in compliance with the License.
  // You may obtain a copy of the License at
  //
  //      http://www.apache.org/licenses/LICENSE-2.0
  //
  // Unless required by applicable law or agreed to in writing, software
  // distributed under the License is distributed on an "AS IS" BASIS,
  // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  // See the License for the specific language governing permissions and
  // limitations under the License.
  var days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
  var months = ['Jan.', 'Feb.', 'March', 'April', 'May', 'June', 'July', 'Aug.', 'Sept.', 'Oct.', 'Nov.', 'Dec.'];
  var zeroPad = function zeroPad(n) {
    return n < 10 ? '0' + n : n;
  };
  var RFC = function RFC(date) {
    var day = days[date.getDay()].substring(0, 3);
    var paddedDate = zeroPad(date.getDate());
    var month = months[date.getMonth()].substring(0, 3);
    var year = date.getFullYear().toString();
    var hours = date.getUTCHours().toString();
    var minutes = date.getUTCMinutes().toString();
    var seconds = date.getUTCSeconds().toString();
    return "".concat(day, ", ").concat(paddedDate, " ").concat(month, " ").concat(year, " ").concat(hours, ":").concat(minutes, ":").concat(seconds, " Z");
  };
  var objectFromMap = function objectFromMap(map) {
    var object = Array.from(map).reduce(function (object, _ref) {
      var _ref2 = _slicedToArray(_ref, 2),
        key = _ref2[0],
        value = _ref2[1];
      return Object.assign(object, _defineProperty({}, key, value)) // Be careful! Maps can have non-String keys; object literals can't.
      ;
    }, {});
    return object;
  };
  var mapFromObject = function mapFromObject(object) {
    var map = new Map();
    for (var property in object) {
      if (object.hasOwnProperty(property)) {
        map.set(property, object[property]);
      }
    }
    return map;
  };
  var Author = /*#__PURE__*/function () {
    // constructor(name='', personalURL='', affiliation='', affiliationURL='') {
    //   this.name = name; // 'Chris Olah'
    //   this.personalURL = personalURL; // 'https://colah.github.io'
    //   this.affiliation = affiliation; // 'Google Brain'
    //   this.affiliationURL = affiliationURL; // 'https://g.co/brain'
    // }

    function Author(object) {
      _classCallCheck(this, Author);
      this.name = object.author; // 'Chris Olah'
      this.personalURL = object.authorURL; // 'https://colah.github.io'
      this.affiliation = object.affiliation; // 'Google Brain'
      this.affiliationURL = object.affiliationURL; // 'https://g.co/brain'
      this.affiliations = object.affiliations || []; // new-style affiliations
    }

    // 'Chris'
    return _createClass(Author, [{
      key: "firstName",
      get: function get() {
        var names = this.name.split(' ');
        return names.slice(0, names.length - 1).join(' ');
      }

      // 'Olah'
    }, {
      key: "lastName",
      get: function get() {
        var names = this.name.split(' ');
        return names[names.length - 1];
      }
    }]);
  }();
  function mergeFromYMLFrontmatter(target, source) {
    target.title = source.title;
    if (source.published) {
      if (source.published instanceof Date) {
        target.publishedDate = source.published;
      } else if (source.published.constructor === String) {
        target.publishedDate = new Date(source.published);
      }
    }
    if (source.publishedDate) {
      if (source.publishedDate instanceof Date) {
        target.publishedDate = source.publishedDate;
      } else if (source.publishedDate.constructor === String) {
        target.publishedDate = new Date(source.publishedDate);
      } else {
        console.error('Don\'t know what to do with published date: ' + source.publishedDate);
      }
    }
    target.description = source.description;
    target.authors = source.authors.map(function (authorObject) {
      return new Author(authorObject);
    });
    target.katex = source.katex;
    target.password = source.password;
    if (source.doi) {
      target.doi = source.doi;
    }
  }
  var FrontMatter = /*#__PURE__*/function () {
    function FrontMatter() {
      _classCallCheck(this, FrontMatter);
      this.title = 'unnamed article'; // 'Attention and Augmented Recurrent Neural Networks'
      this.description = ''; // 'A visual overview of neural attention...'
      this.authors = []; // Array of Author(s)

      this.bibliography = new Map();
      this.bibliographyParsed = false;
      //  {
      //    'gregor2015draw': {
      //      'title': 'DRAW: A recurrent neural network for image generation',
      //      'author': 'Gregor, Karol and Danihelka, Ivo and Graves, Alex and Rezende, Danilo Jimenez and Wierstra, Daan',
      //      'journal': 'arXiv preprint arXiv:1502.04623',
      //      'year': '2015',
      //      'url': 'https://arxiv.org/pdf/1502.04623.pdf',
      //      'type': 'article'
      //    },
      //  }

      // Citation keys should be listed in the order that they are appear in the document.
      // Each key refers to a key in the bibliography dictionary.
      this.citations = []; // [ 'gregor2015draw', 'mercier2011humans' ]
      this.citationsCollected = false;

      //
      // Assigned from posts.csv
      //

      //  publishedDate: 2016-09-08T07:00:00.000Z,
      //  tags: [ 'rnn' ],
      //  distillPath: '2016/augmented-rnns',
      //  githubPath: 'distillpub/post--augmented-rnns',
      //  doiSuffix: 1,

      //
      // Assigned from journal
      //
      this.journal = {};
      //  journal: {
      //    'title': 'Distill',
      //    'full_title': 'Distill',
      //    'abbrev_title': 'Distill',
      //    'url': 'http://distill.pub',
      //    'doi': '10.23915/distill',
      //    'publisherName': 'Distill Working Group',
      //    'publisherEmail': 'admin@distill.pub',
      //    'issn': '2476-0757',
      //    'editors': [...],
      //    'committee': [...]
      //  }
      //  volume: 1,
      //  issue: 9,

      this.katex = {};

      //
      // Assigned from publishing process
      //

      //  githubCompareUpdatesUrl: 'https://github.com/distillpub/post--augmented-rnns/compare/1596e094d8943d2dc0ea445d92071129c6419c59...3bd9209e0c24d020f87cf6152dcecc6017cbc193',
      //  updatedDate: 2017-03-21T07:13:16.000Z,
      //  doi: '10.23915/distill.00001',
      this.doi = undefined;
      this.publishedDate = undefined;
    }

    // Example:
    // title: Demo Title Attention and Augmented Recurrent Neural Networks
    // published: Jan 10, 2017
    // authors:
    // - Chris Olah:
    // - Shan Carter: http://shancarter.com
    // affiliations:
    // - Google Brain:
    // - Google Brain: http://g.co/brain

    //
    // Computed Properties
    //

    // 'http://distill.pub/2016/augmented-rnns',
    return _createClass(FrontMatter, [{
      key: "url",
      get: function get() {
        if (this._url) {
          return this._url;
        } else if (this.distillPath && this.journal.url) {
          return this.journal.url + '/' + this.distillPath;
        } else if (this.journal.url) {
          return this.journal.url;
        }
      }

      // 'https://github.com/distillpub/post--augmented-rnns',
      ,
      set: function set(value) {
        this._url = value;
      }
    }, {
      key: "githubUrl",
      get: function get() {
        if (this.githubPath) {
          return 'https://github.com/' + this.githubPath;
        } else {
          return undefined;
        }
      }

      // TODO resolve differences in naming of URL/Url/url.
      // 'http://distill.pub/2016/augmented-rnns/thumbnail.jpg',
    }, {
      key: "previewURL",
      get: function get() {
        return this._previewURL ? this._previewURL : this.url + '/thumbnail.jpg';
      }

      // 'Thu, 08 Sep 2016 00:00:00 -0700',
      ,
      set: function set(value) {
        this._previewURL = value;
      }
    }, {
      key: "publishedDateRFC",
      get: function get() {
        return RFC(this.publishedDate);
      }

      // 'Thu, 08 Sep 2016 00:00:00 -0700',
    }, {
      key: "updatedDateRFC",
      get: function get() {
        return RFC(this.updatedDate);
      }

      // 2016,
    }, {
      key: "publishedYear",
      get: function get() {
        return this.publishedDate.getFullYear();
      }

      // 'Sept',
    }, {
      key: "publishedMonth",
      get: function get() {
        return months[this.publishedDate.getMonth()];
      }

      // 8,
    }, {
      key: "publishedDay",
      get: function get() {
        return this.publishedDate.getDate();
      }

      // '09',
    }, {
      key: "publishedMonthPadded",
      get: function get() {
        return zeroPad(this.publishedDate.getMonth() + 1);
      }

      // '08',
    }, {
      key: "publishedDayPadded",
      get: function get() {
        return zeroPad(this.publishedDate.getDate());
      }
    }, {
      key: "publishedISODateOnly",
      get: function get() {
        return this.publishedDate.toISOString().split('T')[0];
      }
    }, {
      key: "volume",
      get: function get() {
        var volume = this.publishedYear - 2015;
        if (volume < 1) {
          throw new Error('Invalid publish date detected during computing volume');
        }
        return volume;
      }
    }, {
      key: "issue",
      get: function get() {
        return this.publishedDate.getMonth() + 1;
      }

      // 'Olah & Carter',
    }, {
      key: "concatenatedAuthors",
      get: function get() {
        if (this.authors.length > 2) {
          return this.authors[0].lastName + ', et al.';
        } else if (this.authors.length === 2) {
          return this.authors[0].lastName + ' & ' + this.authors[1].lastName;
        } else if (this.authors.length === 1) {
          return this.authors[0].lastName;
        }
      }

      // 'Olah, Chris and Carter, Shan',
    }, {
      key: "bibtexAuthors",
      get: function get() {
        return this.authors.map(function (author) {
          return author.lastName + ', ' + author.firstName;
        }).join(' and ');
      }

      // 'olah2016attention'
    }, {
      key: "slug",
      get: function get() {
        var slug = '';
        if (this.authors.length) {
          slug += this.authors[0].lastName.toLowerCase();
          slug += this.publishedYear;
          slug += this.title.split(' ')[0].toLowerCase();
        }
        return slug || 'Untitled';
      }
    }, {
      key: "bibliographyEntries",
      get: function get() {
        var _this = this;
        return new Map(this.citations.map(function (citationKey) {
          var entry = _this.bibliography.get(citationKey);
          return [citationKey, entry];
        }));
      }
    }, {
      key: "bibliography",
      get: function get() {
        return this._bibliography;
      },
      set: function set(bibliography) {
        if (bibliography instanceof Map) {
          this._bibliography = bibliography;
        } else if (_typeof(bibliography) === 'object') {
          this._bibliography = mapFromObject(bibliography);
        }
      }
    }, {
      key: "assignToObject",
      value: function assignToObject(target) {
        Object.assign(target, this);
        target.bibliography = objectFromMap(this.bibliographyEntries);
        target.url = this.url;
        target.doi = this.doi;
        target.githubUrl = this.githubUrl;
        target.previewURL = this.previewURL;
        if (this.publishedDate) {
          target.volume = this.volume;
          target.issue = this.issue;
          target.publishedDateRFC = this.publishedDateRFC;
          target.publishedYear = this.publishedYear;
          target.publishedMonth = this.publishedMonth;
          target.publishedDay = this.publishedDay;
          target.publishedMonthPadded = this.publishedMonthPadded;
          target.publishedDayPadded = this.publishedDayPadded;
        }
        if (this.updatedDate) {
          target.updatedDateRFC = this.updatedDateRFC;
        }
        target.concatenatedAuthors = this.concatenatedAuthors;
        target.bibtexAuthors = this.bibtexAuthors;
        target.slug = this.slug;
      }
    }], [{
      key: "fromObject",
      value: function fromObject(source) {
        var frontMatter = new FrontMatter();
        Object.assign(frontMatter, source);
        return frontMatter;
      }
    }]);
  }(); // Copyright 2018 The Distill Template Authors
  //
  // Licensed under the Apache License, Version 2.0 (the "License");
  // you may not use this file except in compliance with the License.
  // You may obtain a copy of the License at
  //
  //      http://www.apache.org/licenses/LICENSE-2.0
  //
  // Unless required by applicable law or agreed to in writing, software
  // distributed under the License is distributed on an "AS IS" BASIS,
  // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  // See the License for the specific language governing permissions and
  // limitations under the License.
  var Mutating = function Mutating(superclass) {
    return /*#__PURE__*/function (_superclass) {
      function _class() {
        var _this2;
        _classCallCheck(this, _class);
        _this2 = _callSuper(this, _class);

        // set up mutation observer
        var options = {
          childList: true,
          characterData: true,
          subtree: true
        };
        var observer = new MutationObserver(function () {
          observer.disconnect();
          _this2.renderIfPossible();
          observer.observe(_this2, options);
        });

        // ...and listen for changes
        observer.observe(_this2, options);
        return _this2;
      }
      _inherits(_class, _superclass);
      return _createClass(_class, [{
        key: "connectedCallback",
        value: function connectedCallback() {
          _superPropGet(_class, "connectedCallback", this, 3)([]);
          this.renderIfPossible();
        }

        // potential TODO: check if this is enough for all our usecases
        // maybe provide a custom function to tell if we have enough information to render
      }, {
        key: "renderIfPossible",
        value: function renderIfPossible() {
          if (this.textContent && this.root) {
            this.renderContent();
          }
        }
      }, {
        key: "renderContent",
        value: function renderContent() {
          console.error("Your class ".concat(this.constructor.name, " must provide a custom renderContent() method!"));
        }
      }]);
    }(superclass); // end class
  }; // end mixin function

  // Copyright 2018 The Distill Template Authors
  //
  // Licensed under the Apache License, Version 2.0 (the "License");
  // you may not use this file except in compliance with the License.
  // You may obtain a copy of the License at
  //
  //      http://www.apache.org/licenses/LICENSE-2.0
  //
  // Unless required by applicable law or agreed to in writing, software
  // distributed under the License is distributed on an "AS IS" BASIS,
  // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  // See the License for the specific language governing permissions and
  // limitations under the License.

  /*global ShadyCSS*/

  var Template = function Template(name, templateString) {
    var useShadow = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : true;
    return function (superclass) {
      var template = document.createElement('template');
      template.innerHTML = templateString;
      if (useShadow && 'ShadyCSS' in window) {
        ShadyCSS.prepareTemplate(template, name);
      }
      return /*#__PURE__*/function (_superclass2) {
        function _class2() {
          var _this3;
          _classCallCheck(this, _class2);
          _this3 = _callSuper(this, _class2);
          _this3.clone = document.importNode(template.content, true);
          if (useShadow) {
            _this3.attachShadow({
              mode: 'open'
            });
            _this3.shadowRoot.appendChild(_this3.clone);
          }
          return _this3;
        }
        _inherits(_class2, _superclass2);
        return _createClass(_class2, [{
          key: "connectedCallback",
          value: function connectedCallback() {
            if (this.hasAttribute('distill-prerendered')) {
              return;
            }
            if (useShadow) {
              if ('ShadyCSS' in window) {
                ShadyCSS.styleElement(this);
              }
            } else {
              this.insertBefore(this.clone, this.firstChild);
            }
          }
        }, {
          key: "root",
          get: function get() {
            if (useShadow) {
              return this.shadowRoot;
            } else {
              return this;
            }
          }

          /* TODO: Are we using these? Should we even? */
        }, {
          key: "$",
          value: function $(query) {
            return this.root.querySelector(query);
          }
        }, {
          key: "$$",
          value: function $$(query) {
            return this.root.querySelectorAll(query);
          }
        }], [{
          key: "is",
          get: function get() {
            return name;
          }
        }]);
      }(superclass);
    };
  };
  var math = "/*\n * Copyright 2018 The Distill Template Authors\n *\n * Licensed under the Apache License, Version 2.0 (the \"License\");\n * you may not use this file except in compliance with the License.\n * You may obtain a copy of the License at\n *\n *      http://www.apache.org/licenses/LICENSE-2.0\n *\n * Unless required by applicable law or agreed to in writing, software\n * distributed under the License is distributed on an \"AS IS\" BASIS,\n * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n * See the License for the specific language governing permissions and\n * limitations under the License.\n */\n\nspan.katex-display {\n  text-align: left;\n  padding: 8px 0 8px 0;\n  margin: 0.5em 0 0.5em 1em;\n}\n\nspan.katex {\n  -webkit-font-smoothing: antialiased;\n  color: rgba(0, 0, 0, 0.8);\n  font-size: 1.18em;\n}\n";

  // Copyright 2018 The Distill Template Authors
  //
  // Licensed under the Apache License, Version 2.0 (the "License");
  // you may not use this file except in compliance with the License.
  // You may obtain a copy of the License at
  //
  //      http://www.apache.org/licenses/LICENSE-2.0
  //
  // Unless required by applicable law or agreed to in writing, software
  // distributed under the License is distributed on an "AS IS" BASIS,
  // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  // See the License for the specific language governing permissions and
  // limitations under the License.

  // This is a straight concatenation of code from KaTeX's contrib folder,
  // but we aren't using some of their helpers that don't work well outside a browser environment.

  /*global katex */

  var findEndOfMath = function findEndOfMath(delimiter, text, startIndex) {
    // Adapted from
    // https://github.com/Khan/perseus/blob/master/src/perseus-markdown.jsx
    var index = startIndex;
    var braceLevel = 0;
    var delimLength = delimiter.length;
    while (index < text.length) {
      var character = text[index];
      if (braceLevel <= 0 && text.slice(index, index + delimLength) === delimiter) {
        return index;
      } else if (character === "\\") {
        index++;
      } else if (character === "{") {
        braceLevel++;
      } else if (character === "}") {
        braceLevel--;
      }
      index++;
    }
    return -1;
  };
  var splitAtDelimiters = function splitAtDelimiters(startData, leftDelim, rightDelim, display) {
    var finalData = [];
    for (var i = 0; i < startData.length; i++) {
      if (startData[i].type === "text") {
        var text = startData[i].data;
        var lookingForLeft = true;
        var currIndex = 0;
        var nextIndex = void 0;
        nextIndex = text.indexOf(leftDelim);
        if (nextIndex !== -1) {
          currIndex = nextIndex;
          finalData.push({
            type: "text",
            data: text.slice(0, currIndex)
          });
          lookingForLeft = false;
        }
        while (true) {
          // eslint-disable-line no-constant-condition
          if (lookingForLeft) {
            nextIndex = text.indexOf(leftDelim, currIndex);
            if (nextIndex === -1) {
              break;
            }
            finalData.push({
              type: "text",
              data: text.slice(currIndex, nextIndex)
            });
            currIndex = nextIndex;
          } else {
            nextIndex = findEndOfMath(rightDelim, text, currIndex + leftDelim.length);
            if (nextIndex === -1) {
              break;
            }
            finalData.push({
              type: "math",
              data: text.slice(currIndex + leftDelim.length, nextIndex),
              rawData: text.slice(currIndex, nextIndex + rightDelim.length),
              display: display
            });
            currIndex = nextIndex + rightDelim.length;
          }
          lookingForLeft = !lookingForLeft;
        }
        finalData.push({
          type: "text",
          data: text.slice(currIndex)
        });
      } else {
        finalData.push(startData[i]);
      }
    }
    return finalData;
  };
  var splitWithDelimiters = function splitWithDelimiters(text, delimiters) {
    var data = [{
      type: "text",
      data: text
    }];
    for (var i = 0; i < delimiters.length; i++) {
      var delimiter = delimiters[i];
      data = splitAtDelimiters(data, delimiter.left, delimiter.right, delimiter.display || false);
    }
    return data;
  };

  /* Note: optionsCopy is mutated by this method. If it is ever exposed in the
   * API, we should copy it before mutating.
   */
  var renderMathInText = function renderMathInText(text, optionsCopy) {
    var data = splitWithDelimiters(text, optionsCopy.delimiters);
    var fragment = document.createDocumentFragment();
    for (var i = 0; i < data.length; i++) {
      if (data[i].type === "text") {
        fragment.appendChild(document.createTextNode(data[i].data));
      } else {
        var tag = document.createElement("d-math");
        var _math = data[i].data;
        // Override any display mode defined in the settings with that
        // defined by the text itself
        optionsCopy.displayMode = data[i].display;
        try {
          tag.textContent = _math;
          if (optionsCopy.displayMode) {
            tag.setAttribute("block", "");
          }
        } catch (e) {
          if (!(e instanceof katex.ParseError)) {
            throw e;
          }
          optionsCopy.errorCallback("KaTeX auto-render: Failed to parse `" + data[i].data + "` with ", e);
          fragment.appendChild(document.createTextNode(data[i].rawData));
          continue;
        }
        fragment.appendChild(tag);
      }
    }
    return fragment;
  };
  var _renderElem = function renderElem(elem, optionsCopy) {
    for (var i = 0; i < elem.childNodes.length; i++) {
      var childNode = elem.childNodes[i];
      if (childNode.nodeType === 3) {
        // Text node
        var text = childNode.textContent;
        if (optionsCopy.mightHaveMath(text)) {
          var frag = renderMathInText(text, optionsCopy);
          i += frag.childNodes.length - 1;
          elem.replaceChild(frag, childNode);
        }
      } else if (childNode.nodeType === 1) {
        // Element node
        var shouldRender = optionsCopy.ignoredTags.indexOf(childNode.nodeName.toLowerCase()) === -1;
        if (shouldRender) {
          _renderElem(childNode, optionsCopy);
        }
      }
      // Otherwise, it's something else, and ignore it.
    }
  };
  var defaultAutoRenderOptions = {
    delimiters: [{
      left: "$$",
      right: "$$",
      display: true
    }, {
      left: "\\[",
      right: "\\]",
      display: true
    }, {
      left: "\\(",
      right: "\\)",
      display: false
    }
    // LaTeX uses this, but it ruins the display of normal `$` in text:
    // {left: '$', right: '$', display: false},
    ],
    ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code", "svg"],
    errorCallback: function errorCallback(msg, err) {
      console.error(msg, err);
    }
  };
  var renderMathInElement = function renderMathInElement(elem, options) {
    if (!elem) {
      throw new Error("No element provided to render");
    }
    var optionsCopy = Object.assign({}, defaultAutoRenderOptions, options);
    var delimiterStrings = optionsCopy.delimiters.flatMap(function (d) {
      return [d.left, d.right];
    });
    var mightHaveMath = function mightHaveMath(text) {
      return delimiterStrings.some(function (d) {
        return text.indexOf(d) !== -1;
      });
    };
    optionsCopy.mightHaveMath = mightHaveMath;
    _renderElem(elem, optionsCopy);
  };

  // Copyright 2018 The Distill Template Authors

  var katexJSURL = 'https://distill.pub/third-party/katex/katex.min.js';
  var katexCSSTag = '<link rel="stylesheet" href="https://distill.pub/third-party/katex/katex.min.css" crossorigin="anonymous">';
  var T = Template('d-math', "\n".concat(katexCSSTag, "\n<style>\n\n:host {\n  display: inline-block;\n  contain: style;\n}\n\n:host([block]) {\n  display: block;\n}\n\n").concat(math, "\n</style>\n<span id='katex-container'></span>\n"));

  // DMath, not Math, because that would conflict with the JS built-in
  var DMath = /*#__PURE__*/function (_Mutating) {
    function DMath() {
      _classCallCheck(this, DMath);
      return _callSuper(this, DMath, arguments);
    }
    _inherits(DMath, _Mutating);
    return _createClass(DMath, [{
      key: "options",
      get: function get() {
        var localOptions = {
          displayMode: this.hasAttribute('block')
        };
        return Object.assign(localOptions, DMath.katexOptions);
      }
    }, {
      key: "connectedCallback",
      value: function connectedCallback() {
        _superPropGet(DMath, "connectedCallback", this, 3)([]);
        if (!DMath.katexAdded) {
          DMath.addKatex();
        }
      }
    }, {
      key: "renderContent",
      value: function renderContent() {
        if (typeof katex !== 'undefined') {
          var container = this.root.querySelector('#katex-container');
          katex.render(this.textContent, container, this.options);
        }
      }
    }], [{
      key: "katexOptions",
      get: function get() {
        if (!DMath._katexOptions) {
          DMath._katexOptions = {
            delimiters: [{
              'left': '$$',
              'right': '$$',
              'display': false
            }]
          };
        }
        return DMath._katexOptions;
      },
      set: function set(options) {
        DMath._katexOptions = options;
        if (DMath.katexOptions.delimiters) {
          if (!DMath.katexAdded) {
            DMath.addKatex();
          } else {
            DMath.katexLoadedCallback();
          }
        }
      }
    }, {
      key: "katexLoadedCallback",
      value: function katexLoadedCallback() {
        // render all d-math tags
        var mathTags = document.querySelectorAll('d-math');
        var _iterator = _createForOfIteratorHelper(mathTags),
          _step;
        try {
          for (_iterator.s(); !(_step = _iterator.n()).done;) {
            var mathTag = _step.value;
            mathTag.renderContent();
          }
          // transform inline delimited math to d-math tags
        } catch (err) {
          _iterator.e(err);
        } finally {
          _iterator.f();
        }
        if (DMath.katexOptions.delimiters) {
          renderMathInElement(document.body, DMath.katexOptions);
        }
      }
    }, {
      key: "addKatex",
      value: function addKatex() {
        // css tag can use this convenience function
        document.head.insertAdjacentHTML('beforeend', katexCSSTag);
        // script tag has to be created to work properly
        var scriptTag = document.createElement('script');
        scriptTag.src = katexJSURL;
        scriptTag.async = true;
        scriptTag.onload = DMath.katexLoadedCallback;
        scriptTag.crossorigin = 'anonymous';
        document.head.appendChild(scriptTag);
        DMath.katexAdded = true;
      }
    }]);
  }(Mutating(T(HTMLElement)));
  DMath.katexAdded = false;
  DMath.inlineMathRendered = false;
  window.DMath = DMath; // TODO: check if this can be removed, or if we should expose a distill global

  // Copyright 2018 The Distill Template Authors
  //
  // Licensed under the Apache License, Version 2.0 (the "License");
  // you may not use this file except in compliance with the License.
  // You may obtain a copy of the License at
  //
  //      http://www.apache.org/licenses/LICENSE-2.0
  //
  // Unless required by applicable law or agreed to in writing, software
  // distributed under the License is distributed on an "AS IS" BASIS,
  // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  // See the License for the specific language governing permissions and
  // limitations under the License.

  function collect_citations() {
    var dom = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : document;
    var citations = new Set();
    var citeTags = dom.querySelectorAll("d-cite");
    var _iterator2 = _createForOfIteratorHelper(citeTags),
      _step2;
    try {
      for (_iterator2.s(); !(_step2 = _iterator2.n()).done;) {
        var tag = _step2.value;
        var keyString = tag.getAttribute("key") || tag.getAttribute("bibtex-key");
        var keys = keyString.split(",").map(function (k) {
          return k.trim();
        });
        var _iterator3 = _createForOfIteratorHelper(keys),
          _step3;
        try {
          for (_iterator3.s(); !(_step3 = _iterator3.n()).done;) {
            var key = _step3.value;
            citations.add(key);
          }
        } catch (err) {
          _iterator3.e(err);
        } finally {
          _iterator3.f();
        }
      }
    } catch (err) {
      _iterator2.e(err);
    } finally {
      _iterator2.f();
    }
    return _toConsumableArray(citations);
  }
  function author_string(ent, template, sep, finalSep) {
    if (ent.author == null) {
      return "";
    }
    var names = ent.author.split(" and ");
    var name_strings = names.map(function (name) {
      name = name.trim();
      if (name.indexOf(",") != -1) {
        var last = name.split(",")[0].trim();
        var firsts = name.split(",")[1];
      } else if (name.indexOf(" ") != -1) {
        var last = name.split(" ").slice(-1)[0].trim();
        var firsts = name.split(" ").slice(0, -1).join(" ");
      } else {
        var last = name.trim();
      }
      var initials = "";
      if (firsts != undefined) {
        initials = firsts.trim().split(" ").map(function (s) {
          return s.trim()[0];
        });
        initials = initials.join(".") + ".";
      }
      return template.replace("${F}", firsts).replace("${L}", last).replace("${I}", initials).trim(); // in case one of first or last was empty
    });
    if (names.length > 1) {
      var str = name_strings.slice(0, names.length - 1).join(sep);
      str += (finalSep || sep) + name_strings[names.length - 1];
      return str;
    } else {
      return name_strings[0];
    }
  }
  function venue_string(ent) {
    var cite = ent.journal || ent.booktitle || "";
    if ("volume" in ent) {
      var issue = ent.issue || ent.number;
      issue = issue != undefined ? "(" + issue + ")" : "";
      cite += ", Vol " + ent.volume + issue;
    }
    if ("pages" in ent) {
      cite += ", pp. " + ent.pages;
    }
    if (cite != "") cite += ". ";
    if ("publisher" in ent) {
      cite += ent.publisher;
      if (cite[cite.length - 1] != ".") cite += ".";
    }
    return cite;
  }
  function link_string(ent) {
    if ("url" in ent) {
      var url = ent.url;
      var arxiv_match = /arxiv\.org\/abs\/([0-9\.]*)/.exec(url);
      if (arxiv_match != null) {
        url = "http://arxiv.org/pdf/".concat(arxiv_match[1], ".pdf");
      }
      if (url.slice(-4) == ".pdf") {
        var label = "PDF";
      } else if (url.slice(-5) == ".html") {
        var label = "HTML";
      }
      return " &ensp;<a href=\"".concat(url, "\">[").concat(label || "link", "]</a>");
    } /* else if ("doi" in ent){
      return ` &ensp;<a href="https://doi.org/${ent.doi}" >[DOI]</a>`;
      }*/else {
      return "";
    }
  }
  function doi_string(ent, new_line) {
    if ("doi" in ent) {
      return "".concat(new_line ? "<br>" : "", " <a href=\"https://doi.org/").concat(ent.doi, "\" style=\"text-decoration:inherit;\">DOI: ").concat(ent.doi, "</a>");
    } else {
      return "";
    }
  }
  function title_string(ent) {
    return '<span class="title">' + ent.title + "</span> ";
  }
  function bibliography_cite(ent, fancy) {
    if (ent) {
      var cite = title_string(ent);
      cite += link_string(ent) + "<br>";
      if (ent.author) {
        cite += author_string(ent, "${L}, ${I}", ", ", " and ");
        if (ent.year || ent.date) {
          cite += ", ";
        }
      }
      if (ent.year || ent.date) {
        cite += (ent.year || ent.date) + ". ";
      } else {
        cite += ". ";
      }
      cite += venue_string(ent);
      cite += doi_string(ent);
      return cite;
      /*var cite =  author_string(ent, "${L}, ${I}", ", ", " and ");
      if (ent.year || ent.date){
        cite += ", " + (ent.year || ent.date) + ". "
      } else {
        cite += ". "
      }
      cite += "<b>" + ent.title + "</b>. ";
      cite += venue_string(ent);
      cite += doi_string(ent);
      cite += link_string(ent);
      return cite*/
    } else {
      return "?";
    }
  }
  function hover_cite(ent) {
    if (ent) {
      var cite = "";
      cite += "<strong>" + ent.title + "</strong>";
      cite += link_string(ent);
      cite += "<br>";
      var a_str = author_string(ent, "${I} ${L}", ", ") + ".";
      var v_str = venue_string(ent).trim() + " " + ent.year + ". " + doi_string(ent, true);
      if ((a_str + v_str).length < Math.min(40, ent.title.length)) {
        cite += a_str + " " + v_str;
      } else {
        cite += a_str + "<br>" + v_str;
      }
      return cite;
    } else {
      return "?";
    }
  }
  function domContentLoaded() {
    return ['interactive', 'complete'].indexOf(document.readyState) !== -1;
  }

  // Copyright 2018 The Distill Template Authors
  //
  // Licensed under the Apache License, Version 2.0 (the "License");
  // you may not use this file except in compliance with the License.
  // You may obtain a copy of the License at
  //
  //      http://www.apache.org/licenses/LICENSE-2.0
  //
  // Unless required by applicable law or agreed to in writing, software
  // distributed under the License is distributed on an "AS IS" BASIS,
  // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  // See the License for the specific language governing permissions and
  // limitations under the License.

  function _moveLegacyAffiliationFormatIntoArray(frontMatter) {
    // authors used to have propoerties "affiliation" and "affiliationURL".
    // We now encourage using an array for affiliations containing objects with
    // properties "name" and "url".
    var _iterator4 = _createForOfIteratorHelper(frontMatter.authors),
      _step4;
    try {
      for (_iterator4.s(); !(_step4 = _iterator4.n()).done;) {
        var author = _step4.value;
        var hasOldStyle = Boolean(author.affiliation);
        var hasNewStyle = Boolean(author.affiliations);
        if (!hasOldStyle) continue;
        if (hasNewStyle) {
          console.warn("Author ".concat(author.author, " has both old-style (\"affiliation\" & \"affiliationURL\") and new style (\"affiliations\") affiliation information!"));
        } else {
          var newAffiliation = {
            "name": author.affiliation
          };
          if (author.affiliationURL) newAffiliation.url = author.affiliationURL;
          author.affiliations = [newAffiliation];
        }
      }
    } catch (err) {
      _iterator4.e(err);
    } finally {
      _iterator4.f();
    }
    return frontMatter;
  }
  function parseFrontmatter(element) {
    var scriptTag = element.firstElementChild;
    if (scriptTag) {
      var type = scriptTag.getAttribute('type');
      if (type.split('/')[1] == 'json') {
        var content = scriptTag.textContent;
        var parsed = JSON.parse(content);
        return _moveLegacyAffiliationFormatIntoArray(parsed);
      } else {
        console.error('Distill only supports JSON frontmatter tags anymore; no more YAML.');
      }
    } else {
      console.error('You added a frontmatter tag but did not provide a script tag with front matter data in it. Please take a look at our templates.');
    }
    return {};
  }
  var FrontMatter$1 = /*#__PURE__*/function (_HTMLElement) {
    function FrontMatter$1() {
      var _this4;
      _classCallCheck(this, FrontMatter$1);
      _this4 = _callSuper(this, FrontMatter$1);
      var options = {
        childList: true,
        characterData: true,
        subtree: true
      };
      var observer = new MutationObserver(function (entries) {
        var _iterator5 = _createForOfIteratorHelper(entries),
          _step5;
        try {
          for (_iterator5.s(); !(_step5 = _iterator5.n()).done;) {
            var entry = _step5.value;
            if (entry.target.nodeName === 'SCRIPT' || entry.type === 'characterData') {
              var data = parseFrontmatter(_this4);
              _this4.notify(data);
            }
          }
        } catch (err) {
          _iterator5.e(err);
        } finally {
          _iterator5.f();
        }
      });
      observer.observe(_this4, options);
      return _this4;
    }
    _inherits(FrontMatter$1, _HTMLElement);
    return _createClass(FrontMatter$1, [{
      key: "notify",
      value: function notify(data) {
        var options = {
          detail: data,
          bubbles: true
        };
        var event = new CustomEvent('onFrontMatterChanged', options);
        document.dispatchEvent(event);
      }
    }], [{
      key: "is",
      get: function get() {
        return 'd-front-matter';
      }
    }]);
  }(/*#__PURE__*/_wrapNativeSuper(HTMLElement)); // Copyright 2018 The Distill Template Authors
  //
  // Licensed under the Apache License, Version 2.0 (the "License");
  // you may not use this file except in compliance with the License.
  // You may obtain a copy of the License at
  //
  //      http://www.apache.org/licenses/LICENSE-2.0
  //
  // Unless required by applicable law or agreed to in writing, software
  // distributed under the License is distributed on an "AS IS" BASIS,
  // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  // See the License for the specific language governing permissions and
  // limitations under the License.
  // no appendix -> add appendix
  // title in front, no h1 -> add it
  // no title in front, h1 -> read and put into frontMatter
  // footnote -> footnote list
  // break up bib
  // if citation, no bib-list -> add citation-list
  // if authors, no byline -> add byline
  function optionalComponents(dom, data) {
    var body = dom.body;
    var article = body.querySelector('d-article');

    // If we don't have an article tag, something weird is going on—giving up.
    if (!article) {
      console.warn('No d-article tag found; skipping adding optional components!');
      return;
    }
    var byline = dom.querySelector('d-byline');
    if (!byline) {
      if (data.authors) {
        byline = dom.createElement('d-byline');
        body.insertBefore(byline, article);
      } else {
        console.warn('No authors found in front matter; please add them before submission!');
      }
    }
    var title = dom.querySelector('d-title');
    if (!title) {
      title = dom.createElement('d-title');
      body.insertBefore(title, byline);
    }
    var h1 = title.querySelector('h1');
    if (!h1) {
      h1 = dom.createElement('h1');
      h1.textContent = data.title;
      title.insertBefore(h1, title.firstChild);
    }
    var hasPassword = typeof data.password !== 'undefined';
    var interstitial = body.querySelector('d-interstitial');
    if (hasPassword && !interstitial) {
      var inBrowser = typeof window !== 'undefined';
      var onLocalhost = inBrowser && window.location.hostname.includes('localhost');
      if (!inBrowser || !onLocalhost) {
        interstitial = dom.createElement('d-interstitial');
        interstitial.password = data.password;
        body.insertBefore(interstitial, body.firstChild);
      }
    } else if (!hasPassword && interstitial) {
      interstitial.parentElement.removeChild(this);
    }
    var appendix = dom.querySelector('d-appendix');
    if (!appendix) {
      appendix = dom.createElement('d-appendix');
      dom.body.appendChild(appendix);
    }
    var footnoteList = dom.querySelector('d-footnote-list');
    if (!footnoteList) {
      footnoteList = dom.createElement('d-footnote-list');
      appendix.appendChild(footnoteList);
    }
    var citationList = dom.querySelector('d-citation-list');
    if (!citationList) {
      citationList = dom.createElement('d-citation-list');
      appendix.appendChild(citationList);
    }
  }

  // Copyright 2018 The Distill Template Authors

  var frontMatter = new FrontMatter();
  var Controller = {
    frontMatter: frontMatter,
    waitingOn: {
      bibliography: [],
      citations: []
    },
    listeners: {
      onCiteKeyCreated: function onCiteKeyCreated(event) {
        var _event$detail = _slicedToArray(event.detail, 2),
          citeTag = _event$detail[0],
          keys = _event$detail[1];

        // ensure we have citations
        if (!frontMatter.citationsCollected) {
          // console.debug('onCiteKeyCreated, but unresolved dependency ("citations"). Enqueing.');
          Controller.waitingOn.citations.push(function () {
            return Controller.listeners.onCiteKeyCreated(event);
          });
          return;
        }

        // ensure we have a loaded bibliography
        if (!frontMatter.bibliographyParsed) {
          // console.debug('onCiteKeyCreated, but unresolved dependency ("bibliography"). Enqueing.');
          Controller.waitingOn.bibliography.push(function () {
            return Controller.listeners.onCiteKeyCreated(event);
          });
          return;
        }
        var numbers = keys.map(function (key) {
          return frontMatter.citations.indexOf(key);
        });
        citeTag.numbers = numbers;
        var entries = keys.map(function (key) {
          return frontMatter.bibliography.get(key);
        });
        citeTag.entries = entries;
      },
      onCiteKeyChanged: function onCiteKeyChanged() {
        // const [citeTag, keys] = event.detail;

        // update citations
        frontMatter.citations = collect_citations();
        frontMatter.citationsCollected = true;
        var _iterator6 = _createForOfIteratorHelper(Controller.waitingOn.citations.slice()),
          _step6;
        try {
          for (_iterator6.s(); !(_step6 = _iterator6.n()).done;) {
            var waitingCallback = _step6.value;
            waitingCallback();
          }

          // update bibliography
        } catch (err) {
          _iterator6.e(err);
        } finally {
          _iterator6.f();
        }
        var citationListTag = document.querySelector("d-citation-list");
        var bibliographyEntries = new Map(frontMatter.citations.map(function (citationKey) {
          return [citationKey, frontMatter.bibliography.get(citationKey)];
        }));
        citationListTag.citations = bibliographyEntries;
        var citeTags = document.querySelectorAll("d-cite");
        var _iterator7 = _createForOfIteratorHelper(citeTags),
          _step7;
        try {
          for (_iterator7.s(); !(_step7 = _iterator7.n()).done;) {
            var citeTag = _step7.value;
            console.log(citeTag);
            var keys = citeTag.keys;
            var numbers = keys.map(function (key) {
              return frontMatter.citations.indexOf(key);
            });
            citeTag.numbers = numbers;
            var entries = keys.map(function (key) {
              return frontMatter.bibliography.get(key);
            });
            citeTag.entries = entries;
          }
        } catch (err) {
          _iterator7.e(err);
        } finally {
          _iterator7.f();
        }
      },
      onCiteKeyRemoved: function onCiteKeyRemoved(event) {
        Controller.listeners.onCiteKeyChanged(event);
      },
      onBibliographyChanged: function onBibliographyChanged(event) {
        var citationListTag = document.querySelector("d-citation-list");
        var bibliography = event.detail;
        frontMatter.bibliography = bibliography;
        frontMatter.bibliographyParsed = true;
        var _iterator8 = _createForOfIteratorHelper(Controller.waitingOn.bibliography.slice()),
          _step8;
        try {
          for (_iterator8.s(); !(_step8 = _iterator8.n()).done;) {
            var waitingCallback = _step8.value;
            waitingCallback();
          }

          // ensure we have citations
        } catch (err) {
          _iterator8.e(err);
        } finally {
          _iterator8.f();
        }
        if (!frontMatter.citationsCollected) {
          Controller.waitingOn.citations.push(function () {
            Controller.listeners.onBibliographyChanged({
              target: event.target,
              detail: event.detail
            });
          });
          return;
        }
        if (citationListTag.hasAttribute("distill-prerendered")) {
          console.debug("Citation list was prerendered; not updating it.");
        } else {
          var entries = new Map(frontMatter.citations.map(function (citationKey) {
            return [citationKey, frontMatter.bibliography.get(citationKey)];
          }));
          citationListTag.citations = entries;
        }
      },
      onFootnoteChanged: function onFootnoteChanged() {
        // const footnote = event.detail;
        //TODO: optimize to only update current footnote
        var footnotesList = document.querySelector("d-footnote-list");
        if (footnotesList) {
          var footnotes = document.querySelectorAll("d-footnote");
          footnotesList.footnotes = footnotes;
        }
      },
      onFrontMatterChanged: function onFrontMatterChanged(event) {
        var data = event.detail;
        mergeFromYMLFrontmatter(frontMatter, data);
        var interstitial = document.querySelector("d-interstitial");
        if (interstitial) {
          if (typeof frontMatter.password !== "undefined") {
            interstitial.password = frontMatter.password;
          } else {
            interstitial.parentElement.removeChild(interstitial);
          }
        }
        var prerendered = document.body.hasAttribute("distill-prerendered");
        if (!prerendered && domContentLoaded()) {
          optionalComponents(document, frontMatter);
          var appendix = document.querySelector("distill-appendix");
          if (appendix) {
            appendix.frontMatter = frontMatter;
          }
          var _byline = document.querySelector("d-byline");
          if (_byline) {
            _byline.frontMatter = frontMatter;
          }
          if (data.katex) {
            DMath.katexOptions = data.katex;
          }
        }
      },
      DOMContentLoaded: function DOMContentLoaded() {
        if (Controller.loaded) {
          console.warn("Controller received DOMContentLoaded but was already loaded!");
          return;
        } else if (!domContentLoaded()) {
          console.warn("Controller received DOMContentLoaded at document.readyState: " + document.readyState + "!");
          return;
        } else {
          Controller.loaded = true;
          console.debug("Runlevel 4: Controller running DOMContentLoaded");
        }
        var frontMatterTag = document.querySelector("d-front-matter");
        if (frontMatterTag) {
          var data = parseFrontmatter(frontMatterTag);
          Controller.listeners.onFrontMatterChanged({
            detail: data
          });
        }

        // Resolving "citations" dependency due to initial DOM load
        frontMatter.citations = collect_citations();
        frontMatter.citationsCollected = true;
        var _iterator9 = _createForOfIteratorHelper(Controller.waitingOn.citations.slice()),
          _step9;
        try {
          for (_iterator9.s(); !(_step9 = _iterator9.n()).done;) {
            var _waitingCallback = _step9.value;
            _waitingCallback();
          }
        } catch (err) {
          _iterator9.e(err);
        } finally {
          _iterator9.f();
        }
        if (frontMatter.bibliographyParsed) {
          var _iterator10 = _createForOfIteratorHelper(Controller.waitingOn.bibliography.slice()),
            _step10;
          try {
            for (_iterator10.s(); !(_step10 = _iterator10.n()).done;) {
              var waitingCallback = _step10.value;
              waitingCallback();
            }
          } catch (err) {
            _iterator10.e(err);
          } finally {
            _iterator10.f();
          }
        }
        var footnotesList = document.querySelector("d-footnote-list");
        if (footnotesList) {
          var footnotes = document.querySelectorAll("d-footnote");
          footnotesList.footnotes = footnotes;
        }
      }
    } // listeners
  }; // Controller

  var base = "/*\n * Copyright 2018 The Distill Template Authors\n *\n * Licensed under the Apache License, Version 2.0 (the \"License\");\n * you may not use this file except in compliance with the License.\n * You may obtain a copy of the License at\n *\n *      http://www.apache.org/licenses/LICENSE-2.0\n *\n * Unless required by applicable law or agreed to in writing, software\n * distributed under the License is distributed on an \"AS IS\" BASIS,\n * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n * See the License for the specific language governing permissions and\n * limitations under the License.\n */\n\nhtml {\n  font-size: 14px;\n\tline-height: 1.6em;\n  /* font-family: \"Libre Franklin\", \"Helvetica Neue\", sans-serif; */\n  font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, Oxygen, Ubuntu, Cantarell, \"Fira Sans\", \"Droid Sans\", \"Helvetica Neue\", Arial, sans-serif;\n  /*, \"Apple Color Emoji\", \"Segoe UI Emoji\", \"Segoe UI Symbol\";*/\n  text-size-adjust: 100%;\n  -ms-text-size-adjust: 100%;\n  -webkit-text-size-adjust: 100%;\n}\n\n@media(min-width: 768px) {\n  html {\n    font-size: 16px;\n  }\n}\n\nbody {\n  margin: 0;\n}\n\na {\n  color: #004276;\n}\n\nfigure {\n  margin: 0;\n}\n\ntable {\n\tborder-collapse: collapse;\n\tborder-spacing: 0;\n}\n\ntable th {\n\ttext-align: left;\n}\n\ntable thead {\n  border-bottom: 1px solid rgba(0, 0, 0, 0.05);\n}\n\ntable thead th {\n  padding-bottom: 0.5em;\n}\n\ntable tbody :first-child td {\n  padding-top: 0.5em;\n}\n\npre {\n  overflow: auto;\n  max-width: 100%;\n}\n\np {\n  margin-top: 0;\n  margin-bottom: 1em;\n}\n\nsup, sub {\n  vertical-align: baseline;\n  position: relative;\n  top: -0.4em;\n  line-height: 1em;\n}\n\nsub {\n  top: 0.4em;\n}\n\n.kicker,\n.marker {\n  font-size: 15px;\n  font-weight: 600;\n  color: rgba(0, 0, 0, 0.5);\n}\n\n\n/* Headline */\n\n@media(min-width: 1024px) {\n  d-title h1 span {\n    display: block;\n  }\n}\n\n/* Figure */\n\nfigure {\n  position: relative;\n  margin-bottom: 2.5em;\n  margin-top: 1.5em;\n}\n\nfigcaption+figure {\n\n}\n\nfigure img {\n  width: 100%;\n}\n\nfigure svg text,\nfigure svg tspan {\n}\n\nfigcaption,\n.figcaption {\n  color: rgba(0, 0, 0, 0.6);\n  font-size: 12px;\n  line-height: 1.5em;\n}\n\n@media(min-width: 1024px) {\nfigcaption,\n.figcaption {\n    font-size: 13px;\n  }\n}\n\nfigure.external img {\n  background: white;\n  border: 1px solid rgba(0, 0, 0, 0.1);\n  box-shadow: 0 1px 8px rgba(0, 0, 0, 0.1);\n  padding: 18px;\n  box-sizing: border-box;\n}\n\nfigcaption a {\n  color: rgba(0, 0, 0, 0.6);\n}\n\nfigcaption b,\nfigcaption strong, {\n  font-weight: 600;\n  color: rgba(0, 0, 0, 1.0);\n}\n";
  var layout = "/*\n * Copyright 2018 The Distill Template Authors\n *\n * Licensed under the Apache License, Version 2.0 (the \"License\");\n * you may not use this file except in compliance with the License.\n * You may obtain a copy of the License at\n *\n *      http://www.apache.org/licenses/LICENSE-2.0\n *\n * Unless required by applicable law or agreed to in writing, software\n * distributed under the License is distributed on an \"AS IS\" BASIS,\n * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n * See the License for the specific language governing permissions and\n * limitations under the License.\n */\n\n@supports not (display: grid) {\n  .base-grid,\n  distill-header,\n  d-title,\n  d-abstract,\n  d-article,\n  d-appendix,\n  distill-appendix,\n  d-byline,\n  d-footnote-list,\n  d-citation-list,\n  distill-footer {\n    display: block;\n    padding: 8px;\n  }\n}\n\n.base-grid,\ndistill-header,\nd-title,\nd-abstract,\nd-article,\nd-appendix,\ndistill-appendix,\nd-byline,\nd-footnote-list,\nd-citation-list,\ndistill-footer {\n  display: grid;\n  justify-items: stretch;\n  grid-template-columns: [screen-start] 8px [page-start kicker-start text-start gutter-start middle-start] 1fr 1fr 1fr 1fr 1fr 1fr 1fr 1fr [text-end page-end gutter-end kicker-end middle-end] 8px [screen-end];\n  grid-column-gap: 8px;\n}\n\n.grid {\n  display: grid;\n  grid-column-gap: 8px;\n}\n\n@media(min-width: 768px) {\n  .base-grid,\n  distill-header,\n  d-title,\n  d-abstract,\n  d-article,\n  d-appendix,\n  distill-appendix,\n  d-byline,\n  d-footnote-list,\n  d-citation-list,\n  distill-footer {\n    grid-template-columns: [screen-start] 1fr [page-start kicker-start middle-start text-start] 45px 45px 45px 45px 45px 45px 45px 45px [ kicker-end text-end gutter-start] 45px [middle-end] 45px [page-end gutter-end] 1fr [screen-end];\n    grid-column-gap: 16px;\n  }\n\n  .grid {\n    grid-column-gap: 16px;\n  }\n}\n\n@media(min-width: 1000px) {\n  .base-grid,\n  distill-header,\n  d-title,\n  d-abstract,\n  d-article,\n  d-appendix,\n  distill-appendix,\n  d-byline,\n  d-footnote-list,\n  d-citation-list,\n  distill-footer {\n    grid-template-columns: [screen-start] 1fr [page-start kicker-start] 50px [middle-start] 50px [text-start kicker-end] 50px 50px 50px 50px 50px 50px 50px 50px [text-end gutter-start] 50px [middle-end] 50px [page-end gutter-end] 1fr [screen-end];\n    grid-column-gap: 16px;\n  }\n\n  .grid {\n    grid-column-gap: 16px;\n  }\n}\n\n@media(min-width: 1180px) {\n  .base-grid,\n  distill-header,\n  d-title,\n  d-abstract,\n  d-article,\n  d-appendix,\n  distill-appendix,\n  d-byline,\n  d-footnote-list,\n  d-citation-list,\n  distill-footer {\n    grid-template-columns: [screen-start] 1fr [page-start kicker-start] 60px [middle-start] 60px [text-start kicker-end] 60px 60px 60px 60px 60px 60px 60px 60px [text-end gutter-start] 60px [middle-end] 60px [page-end gutter-end] 1fr [screen-end];\n    grid-column-gap: 32px;\n  }\n\n  .grid {\n    grid-column-gap: 32px;\n  }\n}\n\n\n\n\n.base-grid {\n  grid-column: screen;\n}\n\n/* .l-body,\nd-article > *  {\n  grid-column: text;\n}\n\n.l-page,\nd-title > *,\nd-figure {\n  grid-column: page;\n} */\n\n.l-gutter {\n  grid-column: gutter;\n}\n\n.l-text,\n.l-body {\n  grid-column: text;\n}\n\n.l-page {\n  grid-column: page;\n}\n\n.l-body-outset {\n  grid-column: middle;\n}\n\n.l-page-outset {\n  grid-column: page;\n}\n\n.l-screen {\n  grid-column: screen;\n}\n\n.l-screen-inset {\n  grid-column: screen;\n  padding-left: 16px;\n  padding-left: 16px;\n}\n\n\n/* Aside */\n\nd-article aside {\n  grid-column: gutter;\n  font-size: 12px;\n  line-height: 1.6em;\n  color: rgba(0, 0, 0, 0.6)\n}\n\n@media(min-width: 768px) {\n  aside {\n    grid-column: gutter;\n  }\n\n  .side {\n    grid-column: gutter;\n  }\n}\n";
  var print = "/*\n * Copyright 2018 The Distill Template Authors\n *\n * Licensed under the Apache License, Version 2.0 (the \"License\");\n * you may not use this file except in compliance with the License.\n * You may obtain a copy of the License at\n *\n *      http://www.apache.org/licenses/LICENSE-2.0\n *\n * Unless required by applicable law or agreed to in writing, software\n * distributed under the License is distributed on an \"AS IS\" BASIS,\n * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n * See the License for the specific language governing permissions and\n * limitations under the License.\n */\n\n@media print {\n\n  @page {\n    size: 8in 11in;\n    @bottom-right {\n      content: counter(page) \" of \" counter(pages);\n    }\n  }\n\n  html {\n    /* no general margins -- CSS Grid takes care of those */\n  }\n\n  p, code {\n    page-break-inside: avoid;\n  }\n\n  h2, h3 {\n    page-break-after: avoid;\n  }\n\n  d-header {\n    visibility: hidden;\n  }\n\n  d-footer {\n    display: none!important;\n  }\n\n}\n";
  var byline = "/*\n * Copyright 2018 The Distill Template Authors\n *\n * Licensed under the Apache License, Version 2.0 (the \"License\");\n * you may not use this file except in compliance with the License.\n * You may obtain a copy of the License at\n *\n *      http://www.apache.org/licenses/LICENSE-2.0\n *\n * Unless required by applicable law or agreed to in writing, software\n * distributed under the License is distributed on an \"AS IS\" BASIS,\n * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n * See the License for the specific language governing permissions and\n * limitations under the License.\n */\n\nd-byline {\n  contain: style;\n  overflow: hidden;\n  border-top: 1px solid rgba(0, 0, 0, 0.1);\n  font-size: 0.8rem;\n  line-height: 1.8em;\n  padding: 1.5rem 0;\n  min-height: 1.8em;\n}\n\n\nd-byline .byline {\n  grid-template-columns: 1fr 1fr;\n  grid-column: text;\n}\n\n@media(min-width: 768px) {\n  d-byline .byline {\n    grid-template-columns: 1fr 1fr 1fr 1fr;\n  }\n}\n\nd-byline .authors-affiliations {\n  grid-column-end: span 2;\n  grid-template-columns: 1fr 1fr;\n  margin-bottom: 1em;\n}\n\n@media(min-width: 768px) {\n  d-byline .authors-affiliations {\n    margin-bottom: 0;\n  }\n}\n\nd-byline h3 {\n  font-size: 0.6rem;\n  font-weight: 400;\n  color: rgba(0, 0, 0, 0.5);\n  margin: 0;\n  text-transform: uppercase;\n}\n\nd-byline p {\n  margin: 0;\n}\n\nd-byline a,\nd-article d-byline a {\n  color: rgba(0, 0, 0, 0.8);\n  text-decoration: none;\n  border-bottom: none;\n}\n\nd-article d-byline a:hover {\n  text-decoration: underline;\n  border-bottom: none;\n}\n\nd-byline p.author {\n  font-weight: 500;\n}\n\nd-byline .affiliations {\n\n}\n";
  var article = "/*\n * Copyright 2018 The Distill Template Authors\n *\n * Licensed under the Apache License, Version 2.0 (the \"License\");\n * you may not use this file except in compliance with the License.\n * You may obtain a copy of the License at\n *\n *      http://www.apache.org/licenses/LICENSE-2.0\n *\n * Unless required by applicable law or agreed to in writing, software\n * distributed under the License is distributed on an \"AS IS\" BASIS,\n * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n * See the License for the specific language governing permissions and\n * limitations under the License.\n */\n\nd-article {\n  contain: layout style;\n  overflow-x: hidden;\n  border-top: 1px solid rgba(0, 0, 0, 0.1);\n  padding-top: 2rem;\n  color: rgba(0, 0, 0, 0.8);\n}\n\nd-article > * {\n  grid-column: text;\n}\n\n@media(min-width: 768px) {\n  d-article {\n    font-size: 16px;\n  }\n}\n\n@media(min-width: 1024px) {\n  d-article {\n    font-size: 1.06rem;\n    line-height: 1.7em;\n  }\n}\n\n\n/* H2 */\n\n\nd-article .marker {\n  text-decoration: none;\n  border: none;\n  counter-reset: section;\n  grid-column: kicker;\n  line-height: 1.7em;\n}\n\nd-article .marker:hover {\n  border: none;\n}\n\nd-article .marker span {\n  padding: 0 3px 4px;\n  border-bottom: 1px solid rgba(0, 0, 0, 0.2);\n  position: relative;\n  top: 4px;\n}\n\nd-article .marker:hover span {\n  color: rgba(0, 0, 0, 0.7);\n  border-bottom: 1px solid rgba(0, 0, 0, 0.7);\n}\n\nd-article h2 {\n  font-weight: 600;\n  font-size: 24px;\n  line-height: 1.25em;\n  margin: 2rem 0 1.5rem 0;\n  border-bottom: 1px solid rgba(0, 0, 0, 0.1);\n  padding-bottom: 1rem;\n}\n\n@media(min-width: 1024px) {\n  d-article h2 {\n    font-size: 36px;\n  }\n}\n\n/* H3 */\n\nd-article h3 {\n  font-weight: 700;\n  font-size: 18px;\n  line-height: 1.4em;\n  margin-bottom: 1em;\n  margin-top: 2em;\n}\n\n@media(min-width: 1024px) {\n  d-article h3 {\n    font-size: 20px;\n  }\n}\n\n/* H4 */\n\nd-article h4 {\n  font-weight: 600;\n  text-transform: uppercase;\n  font-size: 14px;\n  line-height: 1.4em;\n}\n\nd-article a {\n  color: inherit;\n}\n\nd-article p,\nd-article ul,\nd-article ol,\nd-article blockquote {\n  margin-top: 0;\n  margin-bottom: 1em;\n  margin-left: 0;\n  margin-right: 0;\n}\n\nd-article blockquote {\n  border-left: 2px solid rgba(0, 0, 0, 0.2);\n  padding-left: 2em;\n  font-style: italic;\n  color: rgba(0, 0, 0, 0.6);\n}\n\nd-article a {\n  border-bottom: 1px solid rgba(0, 0, 0, 0.4);\n  text-decoration: none;\n}\n\nd-article a:hover {\n  border-bottom: 1px solid rgba(0, 0, 0, 0.8);\n}\n\nd-article .link {\n  text-decoration: underline;\n  cursor: pointer;\n}\n\nd-article ul,\nd-article ol {\n  padding-left: 24px;\n}\n\nd-article li {\n  margin-bottom: 0.2em;\n  margin-left: 0;\n  padding-left: 0;\n}\n\nd-article li:last-child {\n  margin-bottom: 0;\n}\n\nd-article pre {\n  font-size: 14px;\n  margin-bottom: 20px;\n}\n\nd-article hr {\n  grid-column: screen;\n  width: 100%;\n  border: none;\n  border-bottom: 1px solid rgba(0, 0, 0, 0.1);\n  margin-top: 60px;\n  margin-bottom: 60px;\n}\n\nd-article section {\n  margin-top: 60px;\n  margin-bottom: 60px;\n}\n\nd-article span.equation-mimic {\n  font-family: georgia;\n  font-size: 115%;\n  font-style: italic;\n}\n\nd-article > d-code,\nd-article section > d-code  {\n  display: block;\n}\n\nd-article > d-math[block],\nd-article section > d-math[block]  {\n  display: block;\n}\n\n@media (max-width: 768px) {\n  d-article > d-code,\n  d-article section > d-code,\n  d-article > d-math[block],\n  d-article section > d-math[block] {\n      overflow-x: scroll;\n      -ms-overflow-style: none;  // IE 10+\n      overflow: -moz-scrollbars-none;  // Firefox\n  }\n\n  d-article > d-code::-webkit-scrollbar,\n  d-article section > d-code::-webkit-scrollbar,\n  d-article > d-math[block]::-webkit-scrollbar,\n  d-article section > d-math[block]::-webkit-scrollbar {\n    display: none;  // Safari and Chrome\n  }\n}\n\nd-article .citation {\n  color: #668;\n  cursor: pointer;\n}\n\nd-include {\n  width: auto;\n  display: block;\n}\n\nd-figure {\n  contain: layout style;\n}\n\n/* KaTeX */\n\n.katex, .katex-prerendered {\n  contain: style;\n  display: inline-block;\n}\n\n/* Tables */\n\nd-article table {\n  border-collapse: collapse;\n  margin-bottom: 1.5rem;\n  border-bottom: 1px solid rgba(0, 0, 0, 0.2);\n}\n\nd-article table th {\n  border-bottom: 1px solid rgba(0, 0, 0, 0.2);\n}\n\nd-article table td {\n  border-bottom: 1px solid rgba(0, 0, 0, 0.05);\n}\n\nd-article table tr:last-of-type td {\n  border-bottom: none;\n}\n\nd-article table th,\nd-article table td {\n  font-size: 15px;\n  padding: 2px 8px;\n}\n\nd-article table tbody :first-child td {\n  padding-top: 2px;\n}\n";
  var title = "/*\n * Copyright 2018 The Distill Template Authors\n *\n * Licensed under the Apache License, Version 2.0 (the \"License\");\n * you may not use this file except in compliance with the License.\n * You may obtain a copy of the License at\n *\n *      http://www.apache.org/licenses/LICENSE-2.0\n *\n * Unless required by applicable law or agreed to in writing, software\n * distributed under the License is distributed on an \"AS IS\" BASIS,\n * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n * See the License for the specific language governing permissions and\n * limitations under the License.\n */\n\nd-title {\n  padding: 2rem 0 1.5rem;\n  contain: layout style;\n  overflow-x: hidden;\n}\n\n@media(min-width: 768px) {\n  d-title {\n    padding: 4rem 0 1.5rem;\n  }\n}\n\nd-title h1 {\n  grid-column: text;\n  font-size: 40px;\n  font-weight: 700;\n  line-height: 1.1em;\n  margin: 0 0 0.5rem;\n}\n\n@media(min-width: 768px) {\n  d-title h1 {\n    font-size: 50px;\n  }\n}\n\nd-title p {\n  font-weight: 300;\n  font-size: 1.2rem;\n  line-height: 1.55em;\n  grid-column: text;\n}\n\nd-title .status {\n  margin-top: 0px;\n  font-size: 12px;\n  color: #009688;\n  opacity: 0.8;\n  grid-column: kicker;\n}\n\nd-title .status span {\n  line-height: 1;\n  display: inline-block;\n  padding: 6px 0;\n  border-bottom: 1px solid #80cbc4;\n  font-size: 11px;\n  text-transform: uppercase;\n}\n";

  // Copyright 2018 The Distill Template Authors

  var styles = base + layout + title + byline + article + math + print;
  function makeStyleTag(dom) {
    var styleTagId = 'distill-prerendered-styles';
    var prerenderedTag = dom.getElementById(styleTagId);
    if (!prerenderedTag) {
      var styleTag = dom.createElement('style');
      styleTag.id = styleTagId;
      styleTag.type = 'text/css';
      var cssTextTag = dom.createTextNode(styles);
      styleTag.appendChild(cssTextTag);
      var firstScriptTag = dom.head.querySelector('script');
      dom.head.insertBefore(styleTag, firstScriptTag);
    }
  }

  // Copyright 2018 The Distill Template Authors
  //
  // Licensed under the Apache License, Version 2.0 (the "License");
  // you may not use this file except in compliance with the License.
  // You may obtain a copy of the License at
  //
  //      http://www.apache.org/licenses/LICENSE-2.0
  //
  // Unless required by applicable law or agreed to in writing, software
  // distributed under the License is distributed on an "AS IS" BASIS,
  // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  // See the License for the specific language governing permissions and
  // limitations under the License.

  function addPolyfill(polyfill, polyfillLoadedCallback) {
    console.debug('Runlevel 0: Polyfill required: ' + polyfill.name);
    var script = document.createElement('script');
    script.src = polyfill.url;
    script.async = false;
    if (polyfillLoadedCallback) {
      script.onload = function () {
        polyfillLoadedCallback(polyfill);
      };
    }
    script.onerror = function () {
      new Error('Runlevel 0: Polyfills failed to load script ' + polyfill.name);
    };
    document.head.appendChild(script);
  }
  var polyfills = [{
    name: 'WebComponents',
    support: function support() {
      return 'customElements' in window && 'attachShadow' in Element.prototype && 'getRootNode' in Element.prototype && 'content' in document.createElement('template') && 'Promise' in window && 'from' in Array;
    },
    url: 'https://distill.pub/third-party/polyfills/webcomponents-lite.js'
  }, {
    name: 'IntersectionObserver',
    support: function support() {
      return 'IntersectionObserver' in window && 'IntersectionObserverEntry' in window;
    },
    url: 'https://distill.pub/third-party/polyfills/intersection-observer.js'
  }];
  var Polyfills = /*#__PURE__*/function () {
    function Polyfills() {
      _classCallCheck(this, Polyfills);
    }
    return _createClass(Polyfills, null, [{
      key: "browserSupportsAllFeatures",
      value: function browserSupportsAllFeatures() {
        return polyfills.every(function (poly) {
          return poly.support();
        });
      }
    }, {
      key: "load",
      value: function load(callback) {
        // Define an intermediate callback that checks if all is loaded.
        var polyfillLoaded = function polyfillLoaded(polyfill) {
          polyfill.loaded = true;
          console.debug('Runlevel 0: Polyfill has finished loading: ' + polyfill.name);
          // console.debug(window[polyfill.name]);
          if (Polyfills.neededPolyfills.every(function (poly) {
            return poly.loaded;
          })) {
            console.debug('Runlevel 0: All required polyfills have finished loading.');
            console.debug('Runlevel 0->1.');
            window.distillRunlevel = 1;
            callback();
          }
        };
        // Add polyfill script tags
        var _iterator11 = _createForOfIteratorHelper(Polyfills.neededPolyfills),
          _step11;
        try {
          for (_iterator11.s(); !(_step11 = _iterator11.n()).done;) {
            var polyfill = _step11.value;
            addPolyfill(polyfill, polyfillLoaded);
          }
        } catch (err) {
          _iterator11.e(err);
        } finally {
          _iterator11.f();
        }
      }
    }, {
      key: "neededPolyfills",
      get: function get() {
        if (!Polyfills._neededPolyfills) {
          Polyfills._neededPolyfills = polyfills.filter(function (poly) {
            return !poly.support();
          });
        }
        return Polyfills._neededPolyfills;
      }
    }]);
  }(); // Copyright 2018 The Distill Template Authors
  //
  // Licensed under the Apache License, Version 2.0 (the "License");
  // you may not use this file except in compliance with the License.
  // You may obtain a copy of the License at
  //
  //      http://www.apache.org/licenses/LICENSE-2.0
  //
  // Unless required by applicable law or agreed to in writing, software
  // distributed under the License is distributed on an "AS IS" BASIS,
  // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  // See the License for the specific language governing permissions and
  // limitations under the License.
  // const marginSmall = 16;
  // const marginLarge = 3 * marginSmall;
  // const margin = marginSmall + marginLarge;
  // const gutter = marginSmall;
  // const outsetAmount = margin / 2;
  // const numCols = 4;
  // const numGutters = numCols - 1;
  // const columnWidth = (768 - 2 * marginLarge - numGutters * gutter) / numCols;
  //
  // const screenwidth = 768;
  // const pageWidth = screenwidth - 2 * marginLarge;
  // const bodyWidth = pageWidth - columnWidth - gutter;
  function body(selector) {
    return "".concat(selector, " {\n      grid-column: left / text;\n    }\n  ");
  }

  // Copyright 2018 The Distill Template Authors

  var T$1 = Template('d-abstract', "\n<style>\n  :host {\n    font-size: 1.25rem;\n    line-height: 1.6em;\n    color: rgba(0, 0, 0, 0.7);\n    -webkit-font-smoothing: antialiased;\n  }\n\n  ::slotted(p) {\n    margin-top: 0;\n    margin-bottom: 1em;\n    grid-column: text-start / middle-end;\n  }\n  ".concat(body('d-abstract'), "\n</style>\n\n<slot></slot>\n"));
  var Abstract = /*#__PURE__*/function (_T$) {
    function Abstract() {
      _classCallCheck(this, Abstract);
      return _callSuper(this, Abstract, arguments);
    }
    _inherits(Abstract, _T$);
    return _createClass(Abstract);
  }(T$1(HTMLElement)); // Copyright 2018 The Distill Template Authors
  var T$2 = Template('d-appendix', "\n<style>\n\nd-appendix {\n  contain: layout style;\n  font-size: 0.8em;\n  line-height: 1.7em;\n  margin-top: 60px;\n  margin-bottom: 0;\n  border-top: 1px solid rgba(0, 0, 0, 0.1);\n  color: rgba(0,0,0,0.5);\n  padding-top: 60px;\n  padding-bottom: 48px;\n}\n\nd-appendix h3 {\n  grid-column: page-start / text-start;\n  font-size: 15px;\n  font-weight: 500;\n  margin-top: 1em;\n  margin-bottom: 0;\n  color: rgba(0,0,0,0.65);\n}\n\nd-appendix h3 + * {\n  margin-top: 1em;\n}\n\nd-appendix ol {\n  padding: 0 0 0 15px;\n}\n\n@media (min-width: 768px) {\n  d-appendix ol {\n    padding: 0 0 0 30px;\n    margin-left: -30px;\n  }\n}\n\nd-appendix li {\n  margin-bottom: 1em;\n}\n\nd-appendix a {\n  color: rgba(0, 0, 0, 0.6);\n}\n\nd-appendix > * {\n  grid-column: text;\n}\n\nd-appendix > d-footnote-list,\nd-appendix > d-citation-list,\nd-appendix > distill-appendix {\n  grid-column: screen;\n}\n\n</style>\n\n", false);
  var Appendix = /*#__PURE__*/function (_T$2) {
    function Appendix() {
      _classCallCheck(this, Appendix);
      return _callSuper(this, Appendix, arguments);
    }
    _inherits(Appendix, _T$2);
    return _createClass(Appendix);
  }(T$2(HTMLElement)); // Copyright 2018 The Distill Template Authors
  //
  // Licensed under the Apache License, Version 2.0 (the "License");
  // you may not use this file except in compliance with the License.
  // You may obtain a copy of the License at
  //
  //      http://www.apache.org/licenses/LICENSE-2.0
  //
  // Unless required by applicable law or agreed to in writing, software
  // distributed under the License is distributed on an "AS IS" BASIS,
  // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  // See the License for the specific language governing permissions and
  // limitations under the License.
  // import { Template } from '../mixins/template';
  // import { Controller } from '../controller';
  var isOnlyWhitespace = /^\s*$/;
  var Article = /*#__PURE__*/function (_HTMLElement2) {
    function Article() {
      var _this5;
      _classCallCheck(this, Article);
      _this5 = _callSuper(this, Article);
      new MutationObserver(function (mutations) {
        var _iterator12 = _createForOfIteratorHelper(mutations),
          _step12;
        try {
          for (_iterator12.s(); !(_step12 = _iterator12.n()).done;) {
            var mutation = _step12.value;
            var _iterator13 = _createForOfIteratorHelper(mutation.addedNodes),
              _step13;
            try {
              for (_iterator13.s(); !(_step13 = _iterator13.n()).done;) {
                var addedNode = _step13.value;
                switch (addedNode.nodeName) {
                  case '#text':
                    {
                      // usually text nodes are only linebreaks.
                      var text = addedNode.nodeValue;
                      if (!isOnlyWhitespace.test(text)) {
                        console.warn('Use of unwrapped text in distill articles is discouraged as it breaks layout! Please wrap any text in a <span> or <p> tag. We found the following text: ' + text);
                        var wrapper = document.createElement('span');
                        wrapper.innerHTML = addedNode.nodeValue;
                        addedNode.parentNode.insertBefore(wrapper, addedNode);
                        addedNode.parentNode.removeChild(addedNode);
                      }
                    }
                    break;
                }
              }
            } catch (err) {
              _iterator13.e(err);
            } finally {
              _iterator13.f();
            }
          }
        } catch (err) {
          _iterator12.e(err);
        } finally {
          _iterator12.f();
        }
      }).observe(_this5, {
        childList: true
      });
      return _this5;
    }
    _inherits(Article, _HTMLElement2);
    return _createClass(Article, null, [{
      key: "is",
      get: function get() {
        return 'd-article';
      }
    }]);
  }(/*#__PURE__*/_wrapNativeSuper(HTMLElement));
  var commonjsGlobal = typeof globalThis !== 'undefined' ? globalThis : typeof window !== 'undefined' ? window : typeof __webpack_require__.g !== 'undefined' ? __webpack_require__.g : typeof self !== 'undefined' ? self : {};
  function createCommonjsModule(fn, module) {
    return module = {
      exports: {}
    }, fn(module, module.exports), module.exports;
  }
  var bibtexParse = createCommonjsModule(function (module, exports) {
    /* start bibtexParse 0.0.22 */

    //Original work by Henrik Muehe (c) 2010
    //
    //CommonJS port by Mikola Lysenko 2013
    //
    //Port to Browser lib by ORCID / RCPETERS
    //
    //Issues:
    //no comment handling within strings
    //no string concatenation
    //no variable values yet
    //Grammar implemented here:
    //bibtex -> (string | preamble | comment | entry)*;
    //string -> '@STRING' '{' key_equals_value '}';
    //preamble -> '@PREAMBLE' '{' value '}';
    //comment -> '@COMMENT' '{' value '}';
    //entry -> '@' key '{' key ',' key_value_list '}';
    //key_value_list -> key_equals_value (',' key_equals_value)*;
    //key_equals_value -> key '=' value;
    //value -> value_quotes | value_braces | key;
    //value_quotes -> '"' .*? '"'; // not quite
    //value_braces -> '{' .*? '"'; // not quite
    (function (exports) {
      function BibtexParser() {
        this.months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"];
        this.notKey = [',', '{', '}', ' ', '='];
        this.pos = 0;
        this.input = "";
        this.entries = new Array();
        this.currentEntry = "";
        this.setInput = function (t) {
          this.input = t;
        };
        this.getEntries = function () {
          return this.entries;
        };
        this.isWhitespace = function (s) {
          return s == ' ' || s == '\r' || s == '\t' || s == '\n';
        };
        this.match = function (s, canCommentOut) {
          if (canCommentOut == undefined || canCommentOut == null) canCommentOut = true;
          this.skipWhitespace(canCommentOut);
          if (this.input.substring(this.pos, this.pos + s.length) == s) {
            this.pos += s.length;
          } else {
            throw "Token mismatch, expected " + s + ", found " + this.input.substring(this.pos);
          }
          this.skipWhitespace(canCommentOut);
        };
        this.tryMatch = function (s, canCommentOut) {
          if (canCommentOut == undefined || canCommentOut == null) canCommentOut = true;
          this.skipWhitespace(canCommentOut);
          if (this.input.substring(this.pos, this.pos + s.length) == s) {
            return true;
          } else {
            return false;
          }
        };

        /* when search for a match all text can be ignored, not just white space */
        this.matchAt = function () {
          while (this.input.length > this.pos && this.input[this.pos] != '@') {
            this.pos++;
          }
          if (this.input[this.pos] == '@') {
            return true;
          }
          return false;
        };
        this.skipWhitespace = function (canCommentOut) {
          while (this.isWhitespace(this.input[this.pos])) {
            this.pos++;
          }
          if (this.input[this.pos] == "%" && canCommentOut == true) {
            while (this.input[this.pos] != "\n") {
              this.pos++;
            }
            this.skipWhitespace(canCommentOut);
          }
        };
        this.value_braces = function () {
          var bracecount = 0;
          this.match("{", false);
          var start = this.pos;
          var escaped = false;
          while (true) {
            if (!escaped) {
              if (this.input[this.pos] == '}') {
                if (bracecount > 0) {
                  bracecount--;
                } else {
                  var end = this.pos;
                  this.match("}", false);
                  return this.input.substring(start, end);
                }
              } else if (this.input[this.pos] == '{') {
                bracecount++;
              } else if (this.pos >= this.input.length - 1) {
                throw "Unterminated value";
              }
            }
            if (this.input[this.pos] == '\\' && escaped == false) escaped = true;else escaped = false;
            this.pos++;
          }
        };
        this.value_comment = function () {
          var str = '';
          var brcktCnt = 0;
          while (!(this.tryMatch("}", false) && brcktCnt == 0)) {
            str = str + this.input[this.pos];
            if (this.input[this.pos] == '{') brcktCnt++;
            if (this.input[this.pos] == '}') brcktCnt--;
            if (this.pos >= this.input.length - 1) {
              throw "Unterminated value:" + this.input.substring(start);
            }
            this.pos++;
          }
          return str;
        };
        this.value_quotes = function () {
          this.match('"', false);
          var start = this.pos;
          var escaped = false;
          while (true) {
            if (!escaped) {
              if (this.input[this.pos] == '"') {
                var end = this.pos;
                this.match('"', false);
                return this.input.substring(start, end);
              } else if (this.pos >= this.input.length - 1) {
                throw "Unterminated value:" + this.input.substring(start);
              }
            }
            if (this.input[this.pos] == '\\' && escaped == false) escaped = true;else escaped = false;
            this.pos++;
          }
        };
        this.single_value = function () {
          var start = this.pos;
          if (this.tryMatch("{")) {
            return this.value_braces();
          } else if (this.tryMatch('"')) {
            return this.value_quotes();
          } else {
            var k = this.key();
            if (k.match("^[0-9]+$")) return k;else if (this.months.indexOf(k.toLowerCase()) >= 0) return k.toLowerCase();else throw "Value expected:" + this.input.substring(start) + ' for key: ' + k;
          }
        };
        this.value = function () {
          var values = [];
          values.push(this.single_value());
          while (this.tryMatch("#")) {
            this.match("#");
            values.push(this.single_value());
          }
          return values.join("");
        };
        this.key = function () {
          var start = this.pos;
          while (true) {
            if (this.pos >= this.input.length) {
              throw "Runaway key";
            } // а-яА-Я is Cyrillic
            //console.log(this.input[this.pos]);
            if (this.notKey.indexOf(this.input[this.pos]) >= 0) {
              return this.input.substring(start, this.pos);
            } else {
              this.pos++;
            }
          }
        };
        this.key_equals_value = function () {
          var key = this.key();
          if (this.tryMatch("=")) {
            this.match("=");
            var val = this.value();
            return [key, val];
          } else {
            throw "... = value expected, equals sign missing:" + this.input.substring(this.pos);
          }
        };
        this.key_value_list = function () {
          var kv = this.key_equals_value();
          this.currentEntry['entryTags'] = {};
          this.currentEntry['entryTags'][kv[0]] = kv[1];
          while (this.tryMatch(",")) {
            this.match(",");
            // fixes problems with commas at the end of a list
            if (this.tryMatch("}")) {
              break;
            }
            kv = this.key_equals_value();
            this.currentEntry['entryTags'][kv[0]] = kv[1];
          }
        };
        this.entry_body = function (d) {
          this.currentEntry = {};
          this.currentEntry['citationKey'] = this.key();
          this.currentEntry['entryType'] = d.substring(1);
          this.match(",");
          this.key_value_list();
          this.entries.push(this.currentEntry);
        };
        this.directive = function () {
          this.match("@");
          return "@" + this.key();
        };
        this.preamble = function () {
          this.currentEntry = {};
          this.currentEntry['entryType'] = 'PREAMBLE';
          this.currentEntry['entry'] = this.value_comment();
          this.entries.push(this.currentEntry);
        };
        this.comment = function () {
          this.currentEntry = {};
          this.currentEntry['entryType'] = 'COMMENT';
          this.currentEntry['entry'] = this.value_comment();
          this.entries.push(this.currentEntry);
        };
        this.entry = function (d) {
          this.entry_body(d);
        };
        this.bibtex = function () {
          while (this.matchAt()) {
            var d = this.directive();
            this.match("{");
            if (d == "@STRING") {
              this.string();
            } else if (d == "@PREAMBLE") {
              this.preamble();
            } else if (d == "@COMMENT") {
              this.comment();
            } else {
              this.entry(d);
            }
            this.match("}");
          }
        };
      }
      exports.toJSON = function (bibtex) {
        var b = new BibtexParser();
        b.setInput(bibtex);
        b.bibtex();
        return b.entries;
      };

      /* added during hackathon don't hate on me */
      exports.toBibtex = function (json) {
        var out = '';
        for (var i in json) {
          out += "@" + json[i].entryType;
          out += '{';
          if (json[i].citationKey) out += json[i].citationKey + ', ';
          if (json[i].entry) out += json[i].entry;
          if (json[i].entryTags) {
            var tags = '';
            for (var jdx in json[i].entryTags) {
              if (tags.length != 0) tags += ', ';
              tags += jdx + '= {' + json[i].entryTags[jdx] + '}';
            }
            out += tags;
          }
          out += '}\n\n';
        }
        return out;
      };
    })(exports);

    /* end bibtexParse */
  });

  // Copyright 2018 The Distill Template Authors

  function normalizeTag(string) {
    return string.replace(/[\t\n ]+/g, ' ').replace(/{\\["^`.'acu~Hvs]( )?([a-zA-Z])}/g, function (full, x, _char) {
      return _char;
    }).replace(/{\\([a-zA-Z])}/g, function (full, _char2) {
      return _char2;
    });
  }
  function parseBibtex(bibtex) {
    var bibliography = new Map();
    var parsedEntries = bibtexParse.toJSON(bibtex);
    var _iterator14 = _createForOfIteratorHelper(parsedEntries),
      _step14;
    try {
      for (_iterator14.s(); !(_step14 = _iterator14.n()).done;) {
        var entry = _step14.value;
        // normalize tags; note entryTags is an object, not Map
        for (var _i = 0, _Object$entries = Object.entries(entry.entryTags); _i < _Object$entries.length; _i++) {
          var _Object$entries$_i = _slicedToArray(_Object$entries[_i], 2),
            key = _Object$entries$_i[0],
            value = _Object$entries$_i[1];
          entry.entryTags[key.toLowerCase()] = normalizeTag(value);
        }
        entry.entryTags.type = entry.entryType;
        // add to bibliography
        bibliography.set(entry.citationKey, entry.entryTags);
      }
    } catch (err) {
      _iterator14.e(err);
    } finally {
      _iterator14.f();
    }
    return bibliography;
  }
  function serializeFrontmatterToBibtex(frontMatter) {
    return "@article{".concat(frontMatter.slug, ",\n  author = {").concat(frontMatter.bibtexAuthors, "},\n  title = {").concat(frontMatter.title, "},\n  journal = {").concat(frontMatter.journal.title, "},\n  year = {").concat(frontMatter.publishedYear, "},\n  note = {").concat(frontMatter.url, "},\n  doi = {").concat(frontMatter.doi, "}\n}");
  }

  // Copyright 2018 The Distill Template Authors
  var Bibliography = /*#__PURE__*/function (_HTMLElement3) {
    function Bibliography() {
      var _this6;
      _classCallCheck(this, Bibliography);
      _this6 = _callSuper(this, Bibliography);

      // set up mutation observer
      var options = {
        childList: true,
        characterData: true,
        subtree: true
      };
      var observer = new MutationObserver(function (entries) {
        var _iterator15 = _createForOfIteratorHelper(entries),
          _step15;
        try {
          for (_iterator15.s(); !(_step15 = _iterator15.n()).done;) {
            var entry = _step15.value;
            if (entry.target.nodeName === 'SCRIPT' || entry.type === 'characterData') {
              _this6.parseIfPossible();
            }
          }
        } catch (err) {
          _iterator15.e(err);
        } finally {
          _iterator15.f();
        }
      });
      observer.observe(_this6, options);
      return _this6;
    }
    _inherits(Bibliography, _HTMLElement3);
    return _createClass(Bibliography, [{
      key: "connectedCallback",
      value: function connectedCallback() {
        var _this7 = this;
        requestAnimationFrame(function () {
          _this7.parseIfPossible();
        });
      }
    }, {
      key: "parseIfPossible",
      value: function parseIfPossible() {
        var scriptTag = this.querySelector('script');
        if (!scriptTag) return;
        if (scriptTag.type == 'text/bibtex') {
          var newBibtex = scriptTag.textContent;
          if (this.bibtex !== newBibtex) {
            this.bibtex = newBibtex;
            var bibliography = parseBibtex(this.bibtex);
            this.notify(bibliography);
          }
        } else if (scriptTag.type == 'text/json') {
          var _bibliography = new Map(JSON.parse(scriptTag.textContent));
          this.notify(_bibliography);
        } else {
          console.warn('Unsupported bibliography script tag type: ' + scriptTag.type);
        }
      }
    }, {
      key: "notify",
      value: function notify(bibliography) {
        var options = {
          detail: bibliography,
          bubbles: true
        };
        var event = new CustomEvent('onBibliographyChanged', options);
        this.dispatchEvent(event);
      }

      /* observe 'src' attribute */
    }, {
      key: "receivedBibtex",
      value: function receivedBibtex(event) {
        var bibliography = parseBibtex(event.target.response);
        this.notify(bibliography);
      }
    }, {
      key: "attributeChangedCallback",
      value: function attributeChangedCallback(name, oldValue, newValue) {
        var _this8 = this;
        var oReq = new XMLHttpRequest();
        oReq.onload = function (e) {
          return _this8.receivedBibtex(e);
        };
        oReq.onerror = function () {
          return console.warn("Could not load Bibtex! (tried ".concat(newValue, ")"));
        };
        oReq.responseType = 'text';
        oReq.open('GET', newValue, true);
        oReq.send();
      }
    }], [{
      key: "is",
      get: function get() {
        return 'd-bibliography';
      }
    }, {
      key: "observedAttributes",
      get: function get() {
        return ['src'];
      }
    }]);
  }(/*#__PURE__*/_wrapNativeSuper(HTMLElement)); // Copyright 2018 The Distill Template Authors
  //
  // Licensed under the Apache License, Version 2.0 (the "License");
  // you may not use this file except in compliance with the License.
  // You may obtain a copy of the License at
  //
  //      http://www.apache.org/licenses/LICENSE-2.0
  //
  // Unless required by applicable law or agreed to in writing, software
  // distributed under the License is distributed on an "AS IS" BASIS,
  // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  // See the License for the specific language governing permissions and
  // limitations under the License.
  // import style from '../styles/d-byline.css';
  function bylineTemplate(frontMatter) {
    return "\n    <div class=\"byline grid\">\n      <div>\n          <h3>Authors</h3>\n          <div>\n              ".concat(frontMatter.authors.map(function (author, i) {
      return "\n              <span class=\"author\">\n        ".concat(author.personalURL ? "\n          <a class=\"name\" href=\"".concat(author.personalURL, "\">").concat(author.name) + (i + 1 < frontMatter.authors.length ? "," : "") + "</a>" : "\n          <span class=\"name\">".concat(author.name) + (i + 1 < frontMatter.authors.length ? "," : "") + "</span>", "\n      </span>\n              ");
    }).join(''), "\n          </div>\n      </div>\n      <div >\n          <h3>Affiliation</h3>\n          <div><a href=\"https://huggingface.co/\">Hugging Face</a>\n          </div>\n      </div>\n      <div >\n          <h3>Published</h3>\n          <div>Feb 19, 2025</div>\n      </div>\n    </div>\n    <div class=\"side pdf-download\">\n      <a href=\"https://huggingface.co/spaces/nanotron/ultrascale-playbook/resolve/main/The_Ultra-Scale_Playbook_Training_LLMs_on_GPU_Clusters.pdf\">Download PDF\n      <br>\n      <img style=\"width: 32px;\" src=\"../assets/images/256px-PDF.png\" alt=\"PDF\"></a>\n      \n    </div>\n");
  }
  var Byline = /*#__PURE__*/function (_HTMLElement4) {
    function Byline() {
      _classCallCheck(this, Byline);
      return _callSuper(this, Byline, arguments);
    }
    _inherits(Byline, _HTMLElement4);
    return _createClass(Byline, [{
      key: "frontMatter",
      set: function set(frontMatter) {
        this.innerHTML = bylineTemplate(frontMatter);
      }
    }], [{
      key: "is",
      get: function get() {
        return 'd-byline';
      }
    }]);
  }(/*#__PURE__*/_wrapNativeSuper(HTMLElement)); // Copyright 2018 The Distill Template Authors
  var T$3 = Template("d-cite", "\n<style>\n\n:host {\n  display: inline-block;\n}\n\n.citation {\n  color: hsla(206, 90%, 20%, 0.7);\n}\n\n.citation-number {\n  cursor: default;\n  white-space: nowrap;\n  font-family: -apple-system, BlinkMacSystemFont, \"Roboto\", Helvetica, sans-serif;\n  font-size: 75%;\n  color: hsla(206, 90%, 20%, 0.7);\n  display: inline-block;\n  line-height: 1.1em;\n  text-align: center;\n  position: relative;\n  top: -2px;\n  margin: 0 2px;\n}\n\nfigcaption .citation-number {\n  font-size: 11px;\n  font-weight: normal;\n  top: -2px;\n  line-height: 1em;\n}\n\nul {\n  margin: 0;\n  padding: 0;\n  list-style-type: none;\n}\n\nul li {\n  padding: 15px 10px 15px 10px;\n  border-bottom: 1px solid rgba(0,0,0,0.1)\n}\n\nul li:last-of-type {\n  border-bottom: none;\n}\n\n</style>\n\n<d-hover-box id=\"hover-box\"></d-hover-box>\n\n<div id=\"citation-\" class=\"citation\">\n  <span class=\"citation-number\"></span>\n</div>\n");
  var Cite = /*#__PURE__*/function (_T$3) {
    /* Lifecycle */
    function Cite() {
      var _this9;
      _classCallCheck(this, Cite);
      _this9 = _callSuper(this, Cite);
      _this9._numbers = [];
      _this9._entries = [];
      return _this9;
    }
    _inherits(Cite, _T$3);
    return _createClass(Cite, [{
      key: "connectedCallback",
      value: function connectedCallback() {
        var _this10 = this;
        this.outerSpan = this.root.querySelector("#citation-");
        this.innerSpan = this.root.querySelector(".citation-number");
        this.hoverBox = this.root.querySelector("d-hover-box");
        window.customElements.whenDefined("d-hover-box").then(function () {
          _this10.hoverBox.listen(_this10);
        });
        // in case this component got connected after values were set
        if (this.numbers) {
          this.displayNumbers(this.numbers);
        }
        if (this.entries) {
          this.displayEntries(this.entries);
        }
      }

      //TODO This causes an infinite loop on firefox with polyfills.
      // This is only needed for interactive editing so no priority.
      // disconnectedCallback() {
      // const options = { detail: [this, this.keys], bubbles: true };
      // const event = new CustomEvent('onCiteKeyRemoved', options);
      // document.dispatchEvent(event);
      // }

      /* observe 'key' attribute */
    }, {
      key: "attributeChangedCallback",
      value: function attributeChangedCallback(name, oldValue, newValue) {
        var eventName = oldValue ? "onCiteKeyChanged" : "onCiteKeyCreated";
        var keys = newValue.split(",").map(function (k) {
          return k.trim();
        });
        var options = {
          detail: [this, keys],
          bubbles: true
        };
        var event = new CustomEvent(eventName, options);
        document.dispatchEvent(event);
      }
    }, {
      key: "key",
      get: function get() {
        return this.getAttribute("key") || this.getAttribute("bibtex-key");
      },
      set: function set(value) {
        this.setAttribute("key", value);
      }
    }, {
      key: "keys",
      get: function get() {
        var result = this.key.split(",");
        console.log(result);
        return result;
      }

      /* Setters & Rendering */
    }, {
      key: "numbers",
      get: function get() {
        return this._numbers;
      },
      set: function set(numbers) {
        this._numbers = numbers;
        this.displayNumbers(numbers);
      }
    }, {
      key: "displayNumbers",
      value: function displayNumbers(numbers) {
        if (!this.innerSpan) return;
        var numberStrings = numbers.map(function (index) {
          return index == -1 ? "?" : index + 1 + "";
        });
        var textContent = "[" + numberStrings.join(", ") + "]";
        this.innerSpan.textContent = textContent;
      }
    }, {
      key: "entries",
      get: function get() {
        return this._entries;
      },
      set: function set(entries) {
        this._entries = entries;
        this.displayEntries(entries);
      }
    }, {
      key: "displayEntries",
      value: function displayEntries(entries) {
        if (!this.hoverBox) return;
        this.hoverBox.innerHTML = "<ul>\n      ".concat(entries.map(hover_cite).map(function (html) {
          return "<li>".concat(html, "</li>");
        }).join("\n"), "\n    </ul>");
      }
    }], [{
      key: "observedAttributes",
      get: function get() {
        return ["key", "bibtex-key"];
      }
    }]);
  }(T$3(HTMLElement)); // Copyright 2018 The Distill Template Authors
  var styles$1 = "\nd-citation-list {\n  contain: style;\n}\n\nd-citation-list .references {\n  grid-column: text;\n}\n\nd-citation-list .references .title {\n  font-weight: 500;\n}\n";
  function renderCitationList(element, entries) {
    var dom = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : document;
    if (entries.size > 0) {
      element.style.display = '';
      var list = element.querySelector('.references');
      if (list) {
        list.innerHTML = '';
      } else {
        var stylesTag = dom.createElement('style');
        stylesTag.innerHTML = styles$1;
        element.appendChild(stylesTag);
        var heading = dom.createElement('h3');
        heading.id = 'references';
        heading.textContent = 'References';
        element.appendChild(heading);
        list = dom.createElement('ol');
        list.id = 'references-list';
        list.className = 'references';
        element.appendChild(list);
      }
      var _iterator16 = _createForOfIteratorHelper(entries),
        _step16;
      try {
        for (_iterator16.s(); !(_step16 = _iterator16.n()).done;) {
          var _step16$value = _slicedToArray(_step16.value, 2),
            key = _step16$value[0],
            entry = _step16$value[1];
          var listItem = dom.createElement('li');
          listItem.id = key;
          listItem.innerHTML = bibliography_cite(entry);
          list.appendChild(listItem);
        }
      } catch (err) {
        _iterator16.e(err);
      } finally {
        _iterator16.f();
      }
    } else {
      element.style.display = 'none';
    }
  }
  var CitationList = /*#__PURE__*/function (_HTMLElement5) {
    function CitationList() {
      _classCallCheck(this, CitationList);
      return _callSuper(this, CitationList, arguments);
    }
    _inherits(CitationList, _HTMLElement5);
    return _createClass(CitationList, [{
      key: "connectedCallback",
      value: function connectedCallback() {
        if (!this.hasAttribute('distill-prerendered')) {
          this.style.display = 'none';
        }
      }
    }, {
      key: "citations",
      set: function set(citations) {
        renderCitationList(this, citations);
      }
    }], [{
      key: "is",
      get: function get() {
        return 'd-citation-list';
      }
    }]);
  }(/*#__PURE__*/_wrapNativeSuper(HTMLElement));
  var prism = createCommonjsModule(function (module) {
    /* **********************************************
         Begin prism-core.js
    ********************************************** */

    var _self = typeof window !== 'undefined' ? window // if in browser
    : typeof WorkerGlobalScope !== 'undefined' && self instanceof WorkerGlobalScope ? self // if in worker
    : {} // if in node js
    ;

    /**
     * Prism: Lightweight, robust, elegant syntax highlighting
     * MIT license http://www.opensource.org/licenses/mit-license.php/
     * @author Lea Verou http://lea.verou.me
     */

    var Prism = function (_self) {
      // Private helper vars
      var lang = /\blang(?:uage)?-([\w-]+)\b/i;
      var uniqueId = 0;
      var _ = {
        manual: _self.Prism && _self.Prism.manual,
        disableWorkerMessageHandler: _self.Prism && _self.Prism.disableWorkerMessageHandler,
        util: {
          encode: function encode(tokens) {
            if (tokens instanceof Token) {
              return new Token(tokens.type, encode(tokens.content), tokens.alias);
            } else if (Array.isArray(tokens)) {
              return tokens.map(encode);
            } else {
              return tokens.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/\u00a0/g, ' ');
            }
          },
          type: function type(o) {
            return Object.prototype.toString.call(o).slice(8, -1);
          },
          objId: function objId(obj) {
            if (!obj['__id']) {
              Object.defineProperty(obj, '__id', {
                value: ++uniqueId
              });
            }
            return obj['__id'];
          },
          // Deep clone a language definition (e.g. to extend it)
          clone: function deepClone(o, visited) {
            var clone,
              id,
              type = _.util.type(o);
            visited = visited || {};
            switch (type) {
              case 'Object':
                id = _.util.objId(o);
                if (visited[id]) {
                  return visited[id];
                }
                clone = {};
                visited[id] = clone;
                for (var key in o) {
                  if (o.hasOwnProperty(key)) {
                    clone[key] = deepClone(o[key], visited);
                  }
                }
                return clone;
              case 'Array':
                id = _.util.objId(o);
                if (visited[id]) {
                  return visited[id];
                }
                clone = [];
                visited[id] = clone;
                o.forEach(function (v, i) {
                  clone[i] = deepClone(v, visited);
                });
                return clone;
              default:
                return o;
            }
          },
          /**
           * Returns the Prism language of the given element set by a `language-xxxx` or `lang-xxxx` class.
           *
           * If no language is set for the element or the element is `null` or `undefined`, `none` will be returned.
           *
           * @param {Element} element
           * @returns {string}
           */
          getLanguage: function getLanguage(element) {
            while (element && !lang.test(element.className)) {
              element = element.parentElement;
            }
            if (element) {
              return (element.className.match(lang) || [, 'none'])[1].toLowerCase();
            }
            return 'none';
          },
          /**
           * Returns the script element that is currently executing.
           *
           * This does __not__ work for line script element.
           *
           * @returns {HTMLScriptElement | null}
           */
          currentScript: function currentScript() {
            if (typeof document === 'undefined') {
              return null;
            }
            if ('currentScript' in document) {
              return document.currentScript;
            }

            // IE11 workaround
            // we'll get the src of the current script by parsing IE11's error stack trace
            // this will not work for inline scripts

            try {
              throw new Error();
            } catch (err) {
              // Get file src url from stack. Specifically works with the format of stack traces in IE.
              // A stack will look like this:
              //
              // Error
              //    at _.util.currentScript (http://localhost/components/prism-core.js:119:5)
              //    at Global code (http://localhost/components/prism-core.js:606:1)

              var src = (/at [^(\r\n]*\((.*):.+:.+\)$/i.exec(err.stack) || [])[1];
              if (src) {
                var scripts = document.getElementsByTagName('script');
                for (var i in scripts) {
                  if (scripts[i].src == src) {
                    return scripts[i];
                  }
                }
              }
              return null;
            }
          }
        },
        languages: {
          extend: function extend(id, redef) {
            var lang = _.util.clone(_.languages[id]);
            for (var key in redef) {
              lang[key] = redef[key];
            }
            return lang;
          },
          /**
           * Insert a token before another token in a language literal
           * As this needs to recreate the object (we cannot actually insert before keys in object literals),
           * we cannot just provide an object, we need an object and a key.
           * @param inside The key (or language id) of the parent
           * @param before The key to insert before.
           * @param insert Object with the key/value pairs to insert
           * @param root The object that contains `inside`. If equal to Prism.languages, it can be omitted.
           */
          insertBefore: function insertBefore(inside, before, insert, root) {
            root = root || _.languages;
            var grammar = root[inside];
            var ret = {};
            for (var token in grammar) {
              if (grammar.hasOwnProperty(token)) {
                if (token == before) {
                  for (var newToken in insert) {
                    if (insert.hasOwnProperty(newToken)) {
                      ret[newToken] = insert[newToken];
                    }
                  }
                }

                // Do not insert token which also occur in insert. See #1525
                if (!insert.hasOwnProperty(token)) {
                  ret[token] = grammar[token];
                }
              }
            }
            var old = root[inside];
            root[inside] = ret;

            // Update references in other language definitions
            _.languages.DFS(_.languages, function (key, value) {
              if (value === old && key != inside) {
                this[key] = ret;
              }
            });
            return ret;
          },
          // Traverse a language definition with Depth First Search
          DFS: function DFS(o, callback, type, visited) {
            visited = visited || {};
            var objId = _.util.objId;
            for (var i in o) {
              if (o.hasOwnProperty(i)) {
                callback.call(o, i, o[i], type || i);
                var property = o[i],
                  propertyType = _.util.type(property);
                if (propertyType === 'Object' && !visited[objId(property)]) {
                  visited[objId(property)] = true;
                  DFS(property, callback, null, visited);
                } else if (propertyType === 'Array' && !visited[objId(property)]) {
                  visited[objId(property)] = true;
                  DFS(property, callback, i, visited);
                }
              }
            }
          }
        },
        plugins: {},
        highlightAll: function highlightAll(async, callback) {
          _.highlightAllUnder(document, async, callback);
        },
        highlightAllUnder: function highlightAllUnder(container, async, callback) {
          var env = {
            callback: callback,
            container: container,
            selector: 'code[class*="language-"], [class*="language-"] code, code[class*="lang-"], [class*="lang-"] code'
          };
          _.hooks.run('before-highlightall', env);
          env.elements = Array.prototype.slice.apply(env.container.querySelectorAll(env.selector));
          _.hooks.run('before-all-elements-highlight', env);
          for (var i = 0, element; element = env.elements[i++];) {
            _.highlightElement(element, async === true, env.callback);
          }
        },
        highlightElement: function highlightElement(element, async, callback) {
          // Find language
          var language = _.util.getLanguage(element);
          var grammar = _.languages[language];

          // Set language on the element, if not present
          element.className = element.className.replace(lang, '').replace(/\s+/g, ' ') + ' language-' + language;

          // Set language on the parent, for styling
          var parent = element.parentNode;
          if (parent && parent.nodeName.toLowerCase() === 'pre') {
            parent.className = parent.className.replace(lang, '').replace(/\s+/g, ' ') + ' language-' + language;
          }
          var code = element.textContent;
          var env = {
            element: element,
            language: language,
            grammar: grammar,
            code: code
          };
          function insertHighlightedCode(highlightedCode) {
            env.highlightedCode = highlightedCode;
            _.hooks.run('before-insert', env);
            env.element.innerHTML = env.highlightedCode;
            _.hooks.run('after-highlight', env);
            _.hooks.run('complete', env);
            callback && callback.call(env.element);
          }
          _.hooks.run('before-sanity-check', env);
          if (!env.code) {
            _.hooks.run('complete', env);
            callback && callback.call(env.element);
            return;
          }
          _.hooks.run('before-highlight', env);
          if (!env.grammar) {
            insertHighlightedCode(_.util.encode(env.code));
            return;
          }
          if (async && _self.Worker) {
            var worker = new Worker(_.filename);
            worker.onmessage = function (evt) {
              insertHighlightedCode(evt.data);
            };
            worker.postMessage(JSON.stringify({
              language: env.language,
              code: env.code,
              immediateClose: true
            }));
          } else {
            insertHighlightedCode(_.highlight(env.code, env.grammar, env.language));
          }
        },
        highlight: function highlight(text, grammar, language) {
          var env = {
            code: text,
            grammar: grammar,
            language: language
          };
          _.hooks.run('before-tokenize', env);
          env.tokens = _.tokenize(env.code, env.grammar);
          _.hooks.run('after-tokenize', env);
          return Token.stringify(_.util.encode(env.tokens), env.language);
        },
        tokenize: function tokenize(text, grammar) {
          var rest = grammar.rest;
          if (rest) {
            for (var token in rest) {
              grammar[token] = rest[token];
            }
            delete grammar.rest;
          }
          var tokenList = new LinkedList();
          addAfter(tokenList, tokenList.head, text);
          matchGrammar(text, tokenList, grammar, tokenList.head, 0);
          return toArray(tokenList);
        },
        hooks: {
          all: {},
          add: function add(name, callback) {
            var hooks = _.hooks.all;
            hooks[name] = hooks[name] || [];
            hooks[name].push(callback);
          },
          run: function run(name, env) {
            var callbacks = _.hooks.all[name];
            if (!callbacks || !callbacks.length) {
              return;
            }
            for (var i = 0, callback; callback = callbacks[i++];) {
              callback(env);
            }
          }
        },
        Token: Token
      };
      _self.Prism = _;
      function Token(type, content, alias, matchedStr, greedy) {
        this.type = type;
        this.content = content;
        this.alias = alias;
        // Copy of the full string this token was created from
        this.length = (matchedStr || '').length | 0;
        this.greedy = !!greedy;
      }
      Token.stringify = function stringify(o, language) {
        if (typeof o == 'string') {
          return o;
        }
        if (Array.isArray(o)) {
          var s = '';
          o.forEach(function (e) {
            s += stringify(e, language);
          });
          return s;
        }
        var env = {
          type: o.type,
          content: stringify(o.content, language),
          tag: 'span',
          classes: ['token', o.type],
          attributes: {},
          language: language
        };
        var aliases = o.alias;
        if (aliases) {
          if (Array.isArray(aliases)) {
            Array.prototype.push.apply(env.classes, aliases);
          } else {
            env.classes.push(aliases);
          }
        }
        _.hooks.run('wrap', env);
        var attributes = '';
        for (var name in env.attributes) {
          attributes += ' ' + name + '="' + (env.attributes[name] || '').replace(/"/g, '&quot;') + '"';
        }
        return '<' + env.tag + ' class="' + env.classes.join(' ') + '"' + attributes + '>' + env.content + '</' + env.tag + '>';
      };

      /**
       * @param {string} text
       * @param {LinkedList<string | Token>} tokenList
       * @param {any} grammar
       * @param {LinkedListNode<string | Token>} startNode
       * @param {number} startPos
       * @param {boolean} [oneshot=false]
       * @param {string} [target]
       */
      function matchGrammar(text, tokenList, grammar, startNode, startPos, oneshot, target) {
        for (var token in grammar) {
          if (!grammar.hasOwnProperty(token) || !grammar[token]) {
            continue;
          }
          var patterns = grammar[token];
          patterns = Array.isArray(patterns) ? patterns : [patterns];
          for (var j = 0; j < patterns.length; ++j) {
            if (target && target == token + ',' + j) {
              return;
            }
            var pattern = patterns[j],
              inside = pattern.inside,
              lookbehind = !!pattern.lookbehind,
              greedy = !!pattern.greedy,
              lookbehindLength = 0,
              alias = pattern.alias;
            if (greedy && !pattern.pattern.global) {
              // Without the global flag, lastIndex won't work
              var flags = pattern.pattern.toString().match(/[imsuy]*$/)[0];
              pattern.pattern = RegExp(pattern.pattern.source, flags + 'g');
            }
            pattern = pattern.pattern || pattern;
            for (
            // iterate the token list and keep track of the current token/string position
            var currentNode = startNode.next, pos = startPos; currentNode !== tokenList.tail; pos += currentNode.value.length, currentNode = currentNode.next) {
              var str = currentNode.value;
              if (tokenList.length > text.length) {
                // Something went terribly wrong, ABORT, ABORT!
                return;
              }
              if (str instanceof Token) {
                continue;
              }
              var removeCount = 1; // this is the to parameter of removeBetween

              if (greedy && currentNode != tokenList.tail.prev) {
                pattern.lastIndex = pos;
                var match = pattern.exec(text);
                if (!match) {
                  break;
                }
                var from = match.index + (lookbehind && match[1] ? match[1].length : 0);
                var to = match.index + match[0].length;
                var p = pos;

                // find the node that contains the match
                p += currentNode.value.length;
                while (from >= p) {
                  currentNode = currentNode.next;
                  p += currentNode.value.length;
                }
                // adjust pos (and p)
                p -= currentNode.value.length;
                pos = p;

                // the current node is a Token, then the match starts inside another Token, which is invalid
                if (currentNode.value instanceof Token) {
                  continue;
                }

                // find the last node which is affected by this match
                for (var k = currentNode; k !== tokenList.tail && (p < to || typeof k.value === 'string' && !k.prev.value.greedy); k = k.next) {
                  removeCount++;
                  p += k.value.length;
                }
                removeCount--;

                // replace with the new match
                str = text.slice(pos, p);
                match.index -= pos;
              } else {
                pattern.lastIndex = 0;
                var match = pattern.exec(str);
              }
              if (!match) {
                if (oneshot) {
                  break;
                }
                continue;
              }
              if (lookbehind) {
                lookbehindLength = match[1] ? match[1].length : 0;
              }
              var from = match.index + lookbehindLength,
                match = match[0].slice(lookbehindLength),
                to = from + match.length,
                before = str.slice(0, from),
                after = str.slice(to);
              var removeFrom = currentNode.prev;
              if (before) {
                removeFrom = addAfter(tokenList, removeFrom, before);
                pos += before.length;
              }
              removeRange(tokenList, removeFrom, removeCount);
              var wrapped = new Token(token, inside ? _.tokenize(match, inside) : match, alias, match, greedy);
              currentNode = addAfter(tokenList, removeFrom, wrapped);
              if (after) {
                addAfter(tokenList, currentNode, after);
              }
              if (removeCount > 1) matchGrammar(text, tokenList, grammar, currentNode.prev, pos, true, token + ',' + j);
              if (oneshot) break;
            }
          }
        }
      }

      /**
       * @typedef LinkedListNode
       * @property {T} value
       * @property {LinkedListNode<T> | null} prev The previous node.
       * @property {LinkedListNode<T> | null} next The next node.
       * @template T
       */

      /**
       * @template T
       */
      function LinkedList() {
        /** @type {LinkedListNode<T>} */
        var head = {
          value: null,
          prev: null,
          next: null
        };
        /** @type {LinkedListNode<T>} */
        var tail = {
          value: null,
          prev: head,
          next: null
        };
        head.next = tail;

        /** @type {LinkedListNode<T>} */
        this.head = head;
        /** @type {LinkedListNode<T>} */
        this.tail = tail;
        this.length = 0;
      }

      /**
       * Adds a new node with the given value to the list.
       * @param {LinkedList<T>} list
       * @param {LinkedListNode<T>} node
       * @param {T} value
       * @returns {LinkedListNode<T>} The added node.
       * @template T
       */
      function addAfter(list, node, value) {
        // assumes that node != list.tail && values.length >= 0
        var next = node.next;
        var newNode = {
          value: value,
          prev: node,
          next: next
        };
        node.next = newNode;
        next.prev = newNode;
        list.length++;
        return newNode;
      }
      /**
       * Removes `count` nodes after the given node. The given node will not be removed.
       * @param {LinkedList<T>} list
       * @param {LinkedListNode<T>} node
       * @param {number} count
       * @template T
       */
      function removeRange(list, node, count) {
        var next = node.next;
        for (var i = 0; i < count && next !== list.tail; i++) {
          next = next.next;
        }
        node.next = next;
        next.prev = node;
        list.length -= i;
      }
      /**
       * @param {LinkedList<T>} list
       * @returns {T[]}
       * @template T
       */
      function toArray(list) {
        var array = [];
        var node = list.head.next;
        while (node !== list.tail) {
          array.push(node.value);
          node = node.next;
        }
        return array;
      }
      if (!_self.document) {
        if (!_self.addEventListener) {
          // in Node.js
          return _;
        }
        if (!_.disableWorkerMessageHandler) {
          // In worker
          _self.addEventListener('message', function (evt) {
            var message = JSON.parse(evt.data),
              lang = message.language,
              code = message.code,
              immediateClose = message.immediateClose;
            _self.postMessage(_.highlight(code, _.languages[lang], lang));
            if (immediateClose) {
              _self.close();
            }
          }, false);
        }
        return _;
      }

      //Get current script and highlight
      var script = _.util.currentScript();
      if (script) {
        _.filename = script.src;
        if (script.hasAttribute('data-manual')) {
          _.manual = true;
        }
      }
      function highlightAutomaticallyCallback() {
        if (!_.manual) {
          _.highlightAll();
        }
      }
      if (!_.manual) {
        // If the document state is "loading", then we'll use DOMContentLoaded.
        // If the document state is "interactive" and the prism.js script is deferred, then we'll also use the
        // DOMContentLoaded event because there might be some plugins or languages which have also been deferred and they
        // might take longer one animation frame to execute which can create a race condition where only some plugins have
        // been loaded when Prism.highlightAll() is executed, depending on how fast resources are loaded.
        // See https://github.com/PrismJS/prism/issues/2102
        var readyState = document.readyState;
        if (readyState === 'loading' || readyState === 'interactive' && script && script.defer) {
          document.addEventListener('DOMContentLoaded', highlightAutomaticallyCallback);
        } else {
          if (window.requestAnimationFrame) {
            window.requestAnimationFrame(highlightAutomaticallyCallback);
          } else {
            window.setTimeout(highlightAutomaticallyCallback, 16);
          }
        }
      }
      return _;
    }(_self);
    if (module.exports) {
      module.exports = Prism;
    }

    // hack for components to work correctly in node.js
    if (typeof commonjsGlobal !== 'undefined') {
      commonjsGlobal.Prism = Prism;
    }

    /* **********************************************
         Begin prism-markup.js
    ********************************************** */

    Prism.languages.markup = {
      'comment': /<!--[\s\S]*?-->/,
      'prolog': /<\?[\s\S]+?\?>/,
      'doctype': {
        pattern: /<!DOCTYPE(?:[^>"'[\]]|"[^"]*"|'[^']*')+(?:\[(?:(?!<!--)[^"'\]]|"[^"]*"|'[^']*'|<!--[\s\S]*?-->)*\]\s*)?>/i,
        greedy: true
      },
      'cdata': /<!\[CDATA\[[\s\S]*?]]>/i,
      'tag': {
        pattern: /<\/?(?!\d)[^\s>\/=$<%]+(?:\s(?:\s*[^\s>\/=]+(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s'">=]+(?=[\s>]))|(?=[\s/>])))+)?\s*\/?>/i,
        greedy: true,
        inside: {
          'tag': {
            pattern: /^<\/?[^\s>\/]+/i,
            inside: {
              'punctuation': /^<\/?/,
              'namespace': /^[^\s>\/:]+:/
            }
          },
          'attr-value': {
            pattern: /=\s*(?:"[^"]*"|'[^']*'|[^\s'">=]+)/i,
            inside: {
              'punctuation': [/^=/, {
                pattern: /^(\s*)["']|["']$/,
                lookbehind: true
              }]
            }
          },
          'punctuation': /\/?>/,
          'attr-name': {
            pattern: /[^\s>\/]+/,
            inside: {
              'namespace': /^[^\s>\/:]+:/
            }
          }
        }
      },
      'entity': /&#?[\da-z]{1,8};/i
    };
    Prism.languages.markup['tag'].inside['attr-value'].inside['entity'] = Prism.languages.markup['entity'];

    // Plugin to make entity title show the real entity, idea by Roman Komarov
    Prism.hooks.add('wrap', function (env) {
      if (env.type === 'entity') {
        env.attributes['title'] = env.content.replace(/&amp;/, '&');
      }
    });
    Object.defineProperty(Prism.languages.markup.tag, 'addInlined', {
      /**
       * Adds an inlined language to markup.
       *
       * An example of an inlined language is CSS with `<style>` tags.
       *
       * @param {string} tagName The name of the tag that contains the inlined language. This name will be treated as
       * case insensitive.
       * @param {string} lang The language key.
       * @example
       * addInlined('style', 'css');
       */
      value: function addInlined(tagName, lang) {
        var includedCdataInside = {};
        includedCdataInside['language-' + lang] = {
          pattern: /(^<!\[CDATA\[)[\s\S]+?(?=\]\]>$)/i,
          lookbehind: true,
          inside: Prism.languages[lang]
        };
        includedCdataInside['cdata'] = /^<!\[CDATA\[|\]\]>$/i;
        var inside = {
          'included-cdata': {
            pattern: /<!\[CDATA\[[\s\S]*?\]\]>/i,
            inside: includedCdataInside
          }
        };
        inside['language-' + lang] = {
          pattern: /[\s\S]+/,
          inside: Prism.languages[lang]
        };
        var def = {};
        def[tagName] = {
          pattern: RegExp(/(<__[\s\S]*?>)(?:<!\[CDATA\[[\s\S]*?\]\]>\s*|[\s\S])*?(?=<\/__>)/.source.replace(/__/g, function () {
            return tagName;
          }), 'i'),
          lookbehind: true,
          greedy: true,
          inside: inside
        };
        Prism.languages.insertBefore('markup', 'cdata', def);
      }
    });
    Prism.languages.xml = Prism.languages.extend('markup', {});
    Prism.languages.html = Prism.languages.markup;
    Prism.languages.mathml = Prism.languages.markup;
    Prism.languages.svg = Prism.languages.markup;

    /* **********************************************
         Begin prism-css.js
    ********************************************** */

    (function (Prism) {
      var string = /("|')(?:\\(?:\r\n|[\s\S])|(?!\1)[^\\\r\n])*\1/;
      Prism.languages.css = {
        'comment': /\/\*[\s\S]*?\*\//,
        'atrule': {
          pattern: /@[\w-]+[\s\S]*?(?:;|(?=\s*\{))/,
          inside: {
            'rule': /^@[\w-]+/,
            'selector-function-argument': {
              pattern: /(\bselector\s*\((?!\s*\))\s*)(?:[^()]|\((?:[^()]|\([^()]*\))*\))+?(?=\s*\))/,
              lookbehind: true,
              alias: 'selector'
            }
            // See rest below
          }
        },
        'url': {
          pattern: RegExp('url\\((?:' + string.source + '|[^\n\r()]*)\\)', 'i'),
          greedy: true,
          inside: {
            'function': /^url/i,
            'punctuation': /^\(|\)$/
          }
        },
        'selector': RegExp('[^{}\\s](?:[^{};"\']|' + string.source + ')*?(?=\\s*\\{)'),
        'string': {
          pattern: string,
          greedy: true
        },
        'property': /[-_a-z\xA0-\uFFFF][-\w\xA0-\uFFFF]*(?=\s*:)/i,
        'important': /!important\b/i,
        'function': /[-a-z0-9]+(?=\()/i,
        'punctuation': /[(){};:,]/
      };
      Prism.languages.css['atrule'].inside.rest = Prism.languages.css;
      var markup = Prism.languages.markup;
      if (markup) {
        markup.tag.addInlined('style', 'css');
        Prism.languages.insertBefore('inside', 'attr-value', {
          'style-attr': {
            pattern: /\s*style=("|')(?:\\[\s\S]|(?!\1)[^\\])*\1/i,
            inside: {
              'attr-name': {
                pattern: /^\s*style/i,
                inside: markup.tag.inside
              },
              'punctuation': /^\s*=\s*['"]|['"]\s*$/,
              'attr-value': {
                pattern: /.+/i,
                inside: Prism.languages.css
              }
            },
            alias: 'language-css'
          }
        }, markup.tag);
      }
    })(Prism);

    /* **********************************************
         Begin prism-clike.js
    ********************************************** */

    Prism.languages.clike = {
      'comment': [{
        pattern: /(^|[^\\])\/\*[\s\S]*?(?:\*\/|$)/,
        lookbehind: true
      }, {
        pattern: /(^|[^\\:])\/\/.*/,
        lookbehind: true,
        greedy: true
      }],
      'string': {
        pattern: /(["'])(?:\\(?:\r\n|[\s\S])|(?!\1)[^\\\r\n])*\1/,
        greedy: true
      },
      'class-name': {
        pattern: /(\b(?:class|interface|extends|implements|trait|instanceof|new)\s+|\bcatch\s+\()[\w.\\]+/i,
        lookbehind: true,
        inside: {
          'punctuation': /[.\\]/
        }
      },
      'keyword': /\b(?:if|else|while|do|for|return|in|instanceof|function|new|try|throw|catch|finally|null|break|continue)\b/,
      'boolean': /\b(?:true|false)\b/,
      'function': /\w+(?=\()/,
      'number': /\b0x[\da-f]+\b|(?:\b\d+\.?\d*|\B\.\d+)(?:e[+-]?\d+)?/i,
      'operator': /[<>]=?|[!=]=?=?|--?|\+\+?|&&?|\|\|?|[?*/~^%]/,
      'punctuation': /[{}[\];(),.:]/
    };

    /* **********************************************
         Begin prism-javascript.js
    ********************************************** */

    Prism.languages.javascript = Prism.languages.extend('clike', {
      'class-name': [Prism.languages.clike['class-name'], {
        pattern: /(^|[^$\w\xA0-\uFFFF])[_$A-Z\xA0-\uFFFF][$\w\xA0-\uFFFF]*(?=\.(?:prototype|constructor))/,
        lookbehind: true
      }],
      'keyword': [{
        pattern: /((?:^|})\s*)(?:catch|finally)\b/,
        lookbehind: true
      }, {
        pattern: /(^|[^.]|\.\.\.\s*)\b(?:as|async(?=\s*(?:function\b|\(|[$\w\xA0-\uFFFF]|$))|await|break|case|class|const|continue|debugger|default|delete|do|else|enum|export|extends|for|from|function|get|if|implements|import|in|instanceof|interface|let|new|null|of|package|private|protected|public|return|set|static|super|switch|this|throw|try|typeof|undefined|var|void|while|with|yield)\b/,
        lookbehind: true
      }],
      'number': /\b(?:(?:0[xX](?:[\dA-Fa-f](?:_[\dA-Fa-f])?)+|0[bB](?:[01](?:_[01])?)+|0[oO](?:[0-7](?:_[0-7])?)+)n?|(?:\d(?:_\d)?)+n|NaN|Infinity)\b|(?:\b(?:\d(?:_\d)?)+\.?(?:\d(?:_\d)?)*|\B\.(?:\d(?:_\d)?)+)(?:[Ee][+-]?(?:\d(?:_\d)?)+)?/,
      // Allow for all non-ASCII characters (See http://stackoverflow.com/a/2008444)
      'function': /#?[_$a-zA-Z\xA0-\uFFFF][$\w\xA0-\uFFFF]*(?=\s*(?:\.\s*(?:apply|bind|call)\s*)?\()/,
      'operator': /--|\+\+|\*\*=?|=>|&&|\|\||[!=]==|<<=?|>>>?=?|[-+*/%&|^!=<>]=?|\.{3}|\?[.?]?|[~:]/
    });
    Prism.languages.javascript['class-name'][0].pattern = /(\b(?:class|interface|extends|implements|instanceof|new)\s+)[\w.\\]+/;
    Prism.languages.insertBefore('javascript', 'keyword', {
      'regex': {
        pattern: /((?:^|[^$\w\xA0-\uFFFF."'\])\s])\s*)\/(?:\[(?:[^\]\\\r\n]|\\.)*]|\\.|[^/\\\[\r\n])+\/[gimyus]{0,6}(?=(?:\s|\/\*[\s\S]*?\*\/)*(?:$|[\r\n,.;:})\]]|\/\/))/,
        lookbehind: true,
        greedy: true
      },
      // This must be declared before keyword because we use "function" inside the look-forward
      'function-variable': {
        pattern: /#?[_$a-zA-Z\xA0-\uFFFF][$\w\xA0-\uFFFF]*(?=\s*[=:]\s*(?:async\s*)?(?:\bfunction\b|(?:\((?:[^()]|\([^()]*\))*\)|[_$a-zA-Z\xA0-\uFFFF][$\w\xA0-\uFFFF]*)\s*=>))/,
        alias: 'function'
      },
      'parameter': [{
        pattern: /(function(?:\s+[_$A-Za-z\xA0-\uFFFF][$\w\xA0-\uFFFF]*)?\s*\(\s*)(?!\s)(?:[^()]|\([^()]*\))+?(?=\s*\))/,
        lookbehind: true,
        inside: Prism.languages.javascript
      }, {
        pattern: /[_$a-z\xA0-\uFFFF][$\w\xA0-\uFFFF]*(?=\s*=>)/i,
        inside: Prism.languages.javascript
      }, {
        pattern: /(\(\s*)(?!\s)(?:[^()]|\([^()]*\))+?(?=\s*\)\s*=>)/,
        lookbehind: true,
        inside: Prism.languages.javascript
      }, {
        pattern: /((?:\b|\s|^)(?!(?:as|async|await|break|case|catch|class|const|continue|debugger|default|delete|do|else|enum|export|extends|finally|for|from|function|get|if|implements|import|in|instanceof|interface|let|new|null|of|package|private|protected|public|return|set|static|super|switch|this|throw|try|typeof|undefined|var|void|while|with|yield)(?![$\w\xA0-\uFFFF]))(?:[_$A-Za-z\xA0-\uFFFF][$\w\xA0-\uFFFF]*\s*)\(\s*)(?!\s)(?:[^()]|\([^()]*\))+?(?=\s*\)\s*\{)/,
        lookbehind: true,
        inside: Prism.languages.javascript
      }],
      'constant': /\b[A-Z](?:[A-Z_]|\dx?)*\b/
    });
    Prism.languages.insertBefore('javascript', 'string', {
      'template-string': {
        pattern: /`(?:\\[\s\S]|\${(?:[^{}]|{(?:[^{}]|{[^}]*})*})+}|(?!\${)[^\\`])*`/,
        greedy: true,
        inside: {
          'template-punctuation': {
            pattern: /^`|`$/,
            alias: 'string'
          },
          'interpolation': {
            pattern: /((?:^|[^\\])(?:\\{2})*)\${(?:[^{}]|{(?:[^{}]|{[^}]*})*})+}/,
            lookbehind: true,
            inside: {
              'interpolation-punctuation': {
                pattern: /^\${|}$/,
                alias: 'punctuation'
              },
              rest: Prism.languages.javascript
            }
          },
          'string': /[\s\S]+/
        }
      }
    });
    if (Prism.languages.markup) {
      Prism.languages.markup.tag.addInlined('script', 'javascript');
    }
    Prism.languages.js = Prism.languages.javascript;

    /* **********************************************
         Begin prism-file-highlight.js
    ********************************************** */

    (function () {
      if (typeof self === 'undefined' || !self.Prism || !self.document || !document.querySelector) {
        return;
      }

      /**
       * @param {Element} [container=document]
       */
      self.Prism.fileHighlight = function (container) {
        container = container || document;
        var Extensions = {
          'js': 'javascript',
          'py': 'python',
          'rb': 'ruby',
          'ps1': 'powershell',
          'psm1': 'powershell',
          'sh': 'bash',
          'bat': 'batch',
          'h': 'c',
          'tex': 'latex'
        };
        Array.prototype.slice.call(container.querySelectorAll('pre[data-src]')).forEach(function (pre) {
          // ignore if already loaded
          if (pre.hasAttribute('data-src-loaded')) {
            return;
          }

          // load current
          var src = pre.getAttribute('data-src');
          var language,
            parent = pre;
          var lang = /\blang(?:uage)?-([\w-]+)\b/i;
          while (parent && !lang.test(parent.className)) {
            parent = parent.parentNode;
          }
          if (parent) {
            language = (pre.className.match(lang) || [, ''])[1];
          }
          if (!language) {
            var extension = (src.match(/\.(\w+)$/) || [, ''])[1];
            language = Extensions[extension] || extension;
          }
          var code = document.createElement('code');
          code.className = 'language-' + language;
          pre.textContent = '';
          code.textContent = 'Loading…';
          pre.appendChild(code);
          var xhr = new XMLHttpRequest();
          xhr.open('GET', src, true);
          xhr.onreadystatechange = function () {
            if (xhr.readyState == 4) {
              if (xhr.status < 400 && xhr.responseText) {
                code.textContent = xhr.responseText;
                Prism.highlightElement(code);
                // mark as loaded
                pre.setAttribute('data-src-loaded', '');
              } else if (xhr.status >= 400) {
                code.textContent = '✖ Error ' + xhr.status + ' while fetching file: ' + xhr.statusText;
              } else {
                code.textContent = '✖ Error: File does not exist or is empty';
              }
            }
          };
          xhr.send(null);
        });
      };
      document.addEventListener('DOMContentLoaded', function () {
        // execute inside handler, for dropping Event as argument
        self.Prism.fileHighlight();
      });
    })();
  });
  Prism.languages.python = {
    'comment': {
      pattern: /(^|[^\\])#.*/,
      lookbehind: true
    },
    'string-interpolation': {
      pattern: /(?:f|rf|fr)(?:("""|''')[\s\S]+?\1|("|')(?:\\.|(?!\2)[^\\\r\n])*\2)/i,
      greedy: true,
      inside: {
        'interpolation': {
          // "{" <expression> <optional "!s", "!r", or "!a"> <optional ":" format specifier> "}"
          pattern: /((?:^|[^{])(?:{{)*){(?!{)(?:[^{}]|{(?!{)(?:[^{}]|{(?!{)(?:[^{}])+})+})+}/,
          lookbehind: true,
          inside: {
            'format-spec': {
              pattern: /(:)[^:(){}]+(?=}$)/,
              lookbehind: true
            },
            'conversion-option': {
              pattern: /![sra](?=[:}]$)/,
              alias: 'punctuation'
            },
            rest: null
          }
        },
        'string': /[\s\S]+/
      }
    },
    'triple-quoted-string': {
      pattern: /(?:[rub]|rb|br)?("""|''')[\s\S]+?\1/i,
      greedy: true,
      alias: 'string'
    },
    'string': {
      pattern: /(?:[rub]|rb|br)?("|')(?:\\.|(?!\1)[^\\\r\n])*\1/i,
      greedy: true
    },
    'function': {
      pattern: /((?:^|\s)def[ \t]+)[a-zA-Z_]\w*(?=\s*\()/g,
      lookbehind: true
    },
    'class-name': {
      pattern: /(\bclass\s+)\w+/i,
      lookbehind: true
    },
    'decorator': {
      pattern: /(^\s*)@\w+(?:\.\w+)*/im,
      lookbehind: true,
      alias: ['annotation', 'punctuation'],
      inside: {
        'punctuation': /\./
      }
    },
    'keyword': /\b(?:and|as|assert|async|await|break|class|continue|def|del|elif|else|except|exec|finally|for|from|global|if|import|in|is|lambda|nonlocal|not|or|pass|print|raise|return|try|while|with|yield)\b/,
    'builtin': /\b(?:__import__|abs|all|any|apply|ascii|basestring|bin|bool|buffer|bytearray|bytes|callable|chr|classmethod|cmp|coerce|compile|complex|delattr|dict|dir|divmod|enumerate|eval|execfile|file|filter|float|format|frozenset|getattr|globals|hasattr|hash|help|hex|id|input|int|intern|isinstance|issubclass|iter|len|list|locals|long|map|max|memoryview|min|next|object|oct|open|ord|pow|property|range|raw_input|reduce|reload|repr|reversed|round|set|setattr|slice|sorted|staticmethod|str|sum|super|tuple|type|unichr|unicode|vars|xrange|zip)\b/,
    'boolean': /\b(?:True|False|None)\b/,
    'number': /(?:\b(?=\d)|\B(?=\.))(?:0[bo])?(?:(?:\d|0x[\da-f])[\da-f]*\.?\d*|\.\d+)(?:e[+-]?\d+)?j?\b/i,
    'operator': /[-+%=]=?|!=|\*\*?=?|\/\/?=?|<[<=>]?|>[=>]?|[&|^~]/,
    'punctuation': /[{}[\];(),.:]/
  };
  Prism.languages.python['string-interpolation'].inside['interpolation'].inside.rest = Prism.languages.python;
  Prism.languages.py = Prism.languages.python;
  Prism.languages.clike = {
    'comment': [{
      pattern: /(^|[^\\])\/\*[\s\S]*?(?:\*\/|$)/,
      lookbehind: true
    }, {
      pattern: /(^|[^\\:])\/\/.*/,
      lookbehind: true,
      greedy: true
    }],
    'string': {
      pattern: /(["'])(?:\\(?:\r\n|[\s\S])|(?!\1)[^\\\r\n])*\1/,
      greedy: true
    },
    'class-name': {
      pattern: /(\b(?:class|interface|extends|implements|trait|instanceof|new)\s+|\bcatch\s+\()[\w.\\]+/i,
      lookbehind: true,
      inside: {
        'punctuation': /[.\\]/
      }
    },
    'keyword': /\b(?:if|else|while|do|for|return|in|instanceof|function|new|try|throw|catch|finally|null|break|continue)\b/,
    'boolean': /\b(?:true|false)\b/,
    'function': /\w+(?=\()/,
    'number': /\b0x[\da-f]+\b|(?:\b\d+\.?\d*|\B\.\d+)(?:e[+-]?\d+)?/i,
    'operator': /[<>]=?|[!=]=?=?|--?|\+\+?|&&?|\|\|?|[?*/~^%]/,
    'punctuation': /[{}[\];(),.:]/
  };
  Prism.languages.lua = {
    'comment': /^#!.+|--(?:\[(=*)\[[\s\S]*?\]\1\]|.*)/m,
    // \z may be used to skip the following space
    'string': {
      pattern: /(["'])(?:(?!\1)[^\\\r\n]|\\z(?:\r\n|\s)|\\(?:\r\n|[\s\S]))*\1|\[(=*)\[[\s\S]*?\]\2\]/,
      greedy: true
    },
    'number': /\b0x[a-f\d]+\.?[a-f\d]*(?:p[+-]?\d+)?\b|\b\d+(?:\.\B|\.?\d*(?:e[+-]?\d+)?\b)|\B\.\d+(?:e[+-]?\d+)?\b/i,
    'keyword': /\b(?:and|break|do|else|elseif|end|false|for|function|goto|if|in|local|nil|not|or|repeat|return|then|true|until|while)\b/,
    'function': /(?!\d)\w+(?=\s*(?:[({]))/,
    'operator': [/[-+*%^&|#]|\/\/?|<[<=]?|>[>=]?|[=~]=?/, {
      // Match ".." but don't break "..."
      pattern: /(^|[^.])\.\.(?!\.)/,
      lookbehind: true
    }],
    'punctuation': /[\[\](){},;]|\.+|:+/
  };
  (function (Prism) {
    // $ set | grep '^[A-Z][^[:space:]]*=' | cut -d= -f1 | tr '\n' '|'
    // + LC_ALL, RANDOM, REPLY, SECONDS.
    // + make sure PS1..4 are here as they are not always set,
    // - some useless things.
    var envVars = '\\b(?:BASH|BASHOPTS|BASH_ALIASES|BASH_ARGC|BASH_ARGV|BASH_CMDS|BASH_COMPLETION_COMPAT_DIR|BASH_LINENO|BASH_REMATCH|BASH_SOURCE|BASH_VERSINFO|BASH_VERSION|COLORTERM|COLUMNS|COMP_WORDBREAKS|DBUS_SESSION_BUS_ADDRESS|DEFAULTS_PATH|DESKTOP_SESSION|DIRSTACK|DISPLAY|EUID|GDMSESSION|GDM_LANG|GNOME_KEYRING_CONTROL|GNOME_KEYRING_PID|GPG_AGENT_INFO|GROUPS|HISTCONTROL|HISTFILE|HISTFILESIZE|HISTSIZE|HOME|HOSTNAME|HOSTTYPE|IFS|INSTANCE|JOB|LANG|LANGUAGE|LC_ADDRESS|LC_ALL|LC_IDENTIFICATION|LC_MEASUREMENT|LC_MONETARY|LC_NAME|LC_NUMERIC|LC_PAPER|LC_TELEPHONE|LC_TIME|LESSCLOSE|LESSOPEN|LINES|LOGNAME|LS_COLORS|MACHTYPE|MAILCHECK|MANDATORY_PATH|NO_AT_BRIDGE|OLDPWD|OPTERR|OPTIND|ORBIT_SOCKETDIR|OSTYPE|PAPERSIZE|PATH|PIPESTATUS|PPID|PS1|PS2|PS3|PS4|PWD|RANDOM|REPLY|SECONDS|SELINUX_INIT|SESSION|SESSIONTYPE|SESSION_MANAGER|SHELL|SHELLOPTS|SHLVL|SSH_AUTH_SOCK|TERM|UID|UPSTART_EVENTS|UPSTART_INSTANCE|UPSTART_JOB|UPSTART_SESSION|USER|WINDOWID|XAUTHORITY|XDG_CONFIG_DIRS|XDG_CURRENT_DESKTOP|XDG_DATA_DIRS|XDG_GREETER_DATA_DIR|XDG_MENU_PREFIX|XDG_RUNTIME_DIR|XDG_SEAT|XDG_SEAT_PATH|XDG_SESSION_DESKTOP|XDG_SESSION_ID|XDG_SESSION_PATH|XDG_SESSION_TYPE|XDG_VTNR|XMODIFIERS)\\b';
    var insideString = {
      'environment': {
        pattern: RegExp("\\$" + envVars),
        alias: 'constant'
      },
      'variable': [
      // [0]: Arithmetic Environment
      {
        pattern: /\$?\(\([\s\S]+?\)\)/,
        greedy: true,
        inside: {
          // If there is a $ sign at the beginning highlight $(( and )) as variable
          'variable': [{
            pattern: /(^\$\(\([\s\S]+)\)\)/,
            lookbehind: true
          }, /^\$\(\(/],
          'number': /\b0x[\dA-Fa-f]+\b|(?:\b\d+\.?\d*|\B\.\d+)(?:[Ee]-?\d+)?/,
          // Operators according to https://www.gnu.org/software/bash/manual/bashref.html#Shell-Arithmetic
          'operator': /--?|-=|\+\+?|\+=|!=?|~|\*\*?|\*=|\/=?|%=?|<<=?|>>=?|<=?|>=?|==?|&&?|&=|\^=?|\|\|?|\|=|\?|:/,
          // If there is no $ sign at the beginning highlight (( and )) as punctuation
          'punctuation': /\(\(?|\)\)?|,|;/
        }
      },
      // [1]: Command Substitution
      {
        pattern: /\$\((?:\([^)]+\)|[^()])+\)|`[^`]+`/,
        greedy: true,
        inside: {
          'variable': /^\$\(|^`|\)$|`$/
        }
      },
      // [2]: Brace expansion
      {
        pattern: /\$\{[^}]+\}/,
        greedy: true,
        inside: {
          'operator': /:[-=?+]?|[!\/]|##?|%%?|\^\^?|,,?/,
          'punctuation': /[\[\]]/,
          'environment': {
            pattern: RegExp("(\\{)" + envVars),
            lookbehind: true,
            alias: 'constant'
          }
        }
      }, /\$(?:\w+|[#?*!@$])/],
      // Escape sequences from echo and printf's manuals, and escaped quotes.
      'entity': /\\(?:[abceEfnrtv\\"]|O?[0-7]{1,3}|x[0-9a-fA-F]{1,2}|u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8})/
    };
    Prism.languages.bash = {
      'shebang': {
        pattern: /^#!\s*\/.*/,
        alias: 'important'
      },
      'comment': {
        pattern: /(^|[^"{\\$])#.*/,
        lookbehind: true
      },
      'function-name': [
      // a) function foo {
      // b) foo() {
      // c) function foo() {
      // but not “foo {”
      {
        // a) and c)
        pattern: /(\bfunction\s+)\w+(?=(?:\s*\(?:\s*\))?\s*\{)/,
        lookbehind: true,
        alias: 'function'
      }, {
        // b)
        pattern: /\b\w+(?=\s*\(\s*\)\s*\{)/,
        alias: 'function'
      }],
      // Highlight variable names as variables in for and select beginnings.
      'for-or-select': {
        pattern: /(\b(?:for|select)\s+)\w+(?=\s+in\s)/,
        alias: 'variable',
        lookbehind: true
      },
      // Highlight variable names as variables in the left-hand part
      // of assignments (“=” and “+=”).
      'assign-left': {
        pattern: /(^|[\s;|&]|[<>]\()\w+(?=\+?=)/,
        inside: {
          'environment': {
            pattern: RegExp("(^|[\\s;|&]|[<>]\\()" + envVars),
            lookbehind: true,
            alias: 'constant'
          }
        },
        alias: 'variable',
        lookbehind: true
      },
      'string': [
      // Support for Here-documents https://en.wikipedia.org/wiki/Here_document
      {
        pattern: /((?:^|[^<])<<-?\s*)(\w+?)\s*(?:\r?\n|\r)[\s\S]*?(?:\r?\n|\r)\2/,
        lookbehind: true,
        greedy: true,
        inside: insideString
      },
      // Here-document with quotes around the tag
      // → No expansion (so no “inside”).
      {
        pattern: /((?:^|[^<])<<-?\s*)(["'])(\w+)\2\s*(?:\r?\n|\r)[\s\S]*?(?:\r?\n|\r)\3/,
        lookbehind: true,
        greedy: true
      },
      // “Normal” string
      {
        pattern: /(^|[^\\](?:\\\\)*)(["'])(?:\\[\s\S]|\$\([^)]+\)|`[^`]+`|(?!\2)[^\\])*\2/,
        lookbehind: true,
        greedy: true,
        inside: insideString
      }],
      'environment': {
        pattern: RegExp("\\$?" + envVars),
        alias: 'constant'
      },
      'variable': insideString.variable,
      'function': {
        pattern: /(^|[\s;|&]|[<>]\()(?:add|apropos|apt|aptitude|apt-cache|apt-get|aspell|automysqlbackup|awk|basename|bash|bc|bconsole|bg|bzip2|cal|cat|cfdisk|chgrp|chkconfig|chmod|chown|chroot|cksum|clear|cmp|column|comm|cp|cron|crontab|csplit|curl|cut|date|dc|dd|ddrescue|debootstrap|df|diff|diff3|dig|dir|dircolors|dirname|dirs|dmesg|du|egrep|eject|env|ethtool|expand|expect|expr|fdformat|fdisk|fg|fgrep|file|find|fmt|fold|format|free|fsck|ftp|fuser|gawk|git|gparted|grep|groupadd|groupdel|groupmod|groups|grub-mkconfig|gzip|halt|head|hg|history|host|hostname|htop|iconv|id|ifconfig|ifdown|ifup|import|install|ip|jobs|join|kill|killall|less|link|ln|locate|logname|logrotate|look|lpc|lpr|lprint|lprintd|lprintq|lprm|ls|lsof|lynx|make|man|mc|mdadm|mkconfig|mkdir|mke2fs|mkfifo|mkfs|mkisofs|mknod|mkswap|mmv|more|most|mount|mtools|mtr|mutt|mv|nano|nc|netstat|nice|nl|nohup|notify-send|npm|nslookup|op|open|parted|passwd|paste|pathchk|ping|pkill|pnpm|popd|pr|printcap|printenv|ps|pushd|pv|quota|quotacheck|quotactl|ram|rar|rcp|reboot|remsync|rename|renice|rev|rm|rmdir|rpm|rsync|scp|screen|sdiff|sed|sendmail|seq|service|sftp|sh|shellcheck|shuf|shutdown|sleep|slocate|sort|split|ssh|stat|strace|su|sudo|sum|suspend|swapon|sync|tac|tail|tar|tee|time|timeout|top|touch|tr|traceroute|tsort|tty|umount|uname|unexpand|uniq|units|unrar|unshar|unzip|update-grub|uptime|useradd|userdel|usermod|users|uudecode|uuencode|v|vdir|vi|vim|virsh|vmstat|wait|watch|wc|wget|whereis|which|who|whoami|write|xargs|xdg-open|yarn|yes|zenity|zip|zsh|zypper)(?=$|[)\s;|&])/,
        lookbehind: true
      },
      'keyword': {
        pattern: /(^|[\s;|&]|[<>]\()(?:if|then|else|elif|fi|for|while|in|case|esac|function|select|do|done|until)(?=$|[)\s;|&])/,
        lookbehind: true
      },
      // https://www.gnu.org/software/bash/manual/html_node/Shell-Builtin-Commands.html
      'builtin': {
        pattern: /(^|[\s;|&]|[<>]\()(?:\.|:|break|cd|continue|eval|exec|exit|export|getopts|hash|pwd|readonly|return|shift|test|times|trap|umask|unset|alias|bind|builtin|caller|command|declare|echo|enable|help|let|local|logout|mapfile|printf|read|readarray|source|type|typeset|ulimit|unalias|set|shopt)(?=$|[)\s;|&])/,
        lookbehind: true,
        // Alias added to make those easier to distinguish from strings.
        alias: 'class-name'
      },
      'boolean': {
        pattern: /(^|[\s;|&]|[<>]\()(?:true|false)(?=$|[)\s;|&])/,
        lookbehind: true
      },
      'file-descriptor': {
        pattern: /\B&\d\b/,
        alias: 'important'
      },
      'operator': {
        // Lots of redirections here, but not just that.
        pattern: /\d?<>|>\||\+=|==?|!=?|=~|<<[<-]?|[&\d]?>>|\d?[<>]&?|&[>&]?|\|[&|]?|<=?|>=?/,
        inside: {
          'file-descriptor': {
            pattern: /^\d/,
            alias: 'important'
          }
        }
      },
      'punctuation': /\$?\(\(?|\)\)?|\.\.|[{}[\];\\]/,
      'number': {
        pattern: /(^|\s)(?:[1-9]\d*|0)(?:[.,]\d+)?\b/,
        lookbehind: true
      }
    };

    /* Patterns in command substitution. */
    var toBeCopied = ['comment', 'function-name', 'for-or-select', 'assign-left', 'string', 'environment', 'function', 'keyword', 'builtin', 'boolean', 'file-descriptor', 'operator', 'punctuation', 'number'];
    var inside = insideString.variable[1].inside;
    for (var i = 0; i < toBeCopied.length; i++) {
      inside[toBeCopied[i]] = Prism.languages.bash[toBeCopied[i]];
    }
    Prism.languages.shell = Prism.languages.bash;
  })(Prism);
  Prism.languages.go = Prism.languages.extend('clike', {
    'keyword': /\b(?:break|case|chan|const|continue|default|defer|else|fallthrough|for|func|go(?:to)?|if|import|interface|map|package|range|return|select|struct|switch|type|var)\b/,
    'builtin': /\b(?:bool|byte|complex(?:64|128)|error|float(?:32|64)|rune|string|u?int(?:8|16|32|64)?|uintptr|append|cap|close|complex|copy|delete|imag|len|make|new|panic|print(?:ln)?|real|recover)\b/,
    'boolean': /\b(?:_|iota|nil|true|false)\b/,
    'operator': /[*\/%^!=]=?|\+[=+]?|-[=-]?|\|[=|]?|&(?:=|&|\^=?)?|>(?:>=?|=)?|<(?:<=?|=|-)?|:=|\.\.\./,
    'number': /(?:\b0x[a-f\d]+|(?:\b\d+\.?\d*|\B\.\d+)(?:e[-+]?\d+)?)i?/i,
    'string': {
      pattern: /(["'`])(?:\\[\s\S]|(?!\1)[^\\])*\1/,
      greedy: true
    }
  });
  delete Prism.languages.go['class-name'];
  (function (Prism) {
    // Allow only one line break
    var inner = /(?:\\.|[^\\\n\r]|(?:\n|\r\n?)(?!\n|\r\n?))/.source;

    /**
     * This function is intended for the creation of the bold or italic pattern.
     *
     * This also adds a lookbehind group to the given pattern to ensure that the pattern is not backslash-escaped.
     *
     * _Note:_ Keep in mind that this adds a capturing group.
     *
     * @param {string} pattern
     * @param {boolean} starAlternative Whether to also add an alternative where all `_`s are replaced with `*`s.
     * @returns {RegExp}
     */
    function createInline(pattern, starAlternative) {
      pattern = pattern.replace(/<inner>/g, function () {
        return inner;
      });
      if (starAlternative) {
        pattern = pattern + '|' + pattern.replace(/_/g, '\\*');
      }
      return RegExp(/((?:^|[^\\])(?:\\{2})*)/.source + '(?:' + pattern + ')');
    }
    var tableCell = /(?:\\.|``.+?``|`[^`\r\n]+`|[^\\|\r\n`])+/.source;
    var tableRow = /\|?__(?:\|__)+\|?(?:(?:\n|\r\n?)|$)/.source.replace(/__/g, function () {
      return tableCell;
    });
    var tableLine = /\|?[ \t]*:?-{3,}:?[ \t]*(?:\|[ \t]*:?-{3,}:?[ \t]*)+\|?(?:\n|\r\n?)/.source;
    Prism.languages.markdown = Prism.languages.extend('markup', {});
    Prism.languages.insertBefore('markdown', 'prolog', {
      'blockquote': {
        // > ...
        pattern: /^>(?:[\t ]*>)*/m,
        alias: 'punctuation'
      },
      'table': {
        pattern: RegExp('^' + tableRow + tableLine + '(?:' + tableRow + ')*', 'm'),
        inside: {
          'table-data-rows': {
            pattern: RegExp('^(' + tableRow + tableLine + ')(?:' + tableRow + ')*$'),
            lookbehind: true,
            inside: {
              'table-data': {
                pattern: RegExp(tableCell),
                inside: Prism.languages.markdown
              },
              'punctuation': /\|/
            }
          },
          'table-line': {
            pattern: RegExp('^(' + tableRow + ')' + tableLine + '$'),
            lookbehind: true,
            inside: {
              'punctuation': /\||:?-{3,}:?/
            }
          },
          'table-header-row': {
            pattern: RegExp('^' + tableRow + '$'),
            inside: {
              'table-header': {
                pattern: RegExp(tableCell),
                alias: 'important',
                inside: Prism.languages.markdown
              },
              'punctuation': /\|/
            }
          }
        }
      },
      'code': [{
        // Prefixed by 4 spaces or 1 tab and preceded by an empty line
        pattern: /((?:^|\n)[ \t]*\n|(?:^|\r\n?)[ \t]*\r\n?)(?: {4}|\t).+(?:(?:\n|\r\n?)(?: {4}|\t).+)*/,
        lookbehind: true,
        alias: 'keyword'
      }, {
        // `code`
        // ``code``
        pattern: /``.+?``|`[^`\r\n]+`/,
        alias: 'keyword'
      }, {
        // ```optional language
        // code block
        // ```
        pattern: /^```[\s\S]*?^```$/m,
        greedy: true,
        inside: {
          'code-block': {
            pattern: /^(```.*(?:\n|\r\n?))[\s\S]+?(?=(?:\n|\r\n?)^```$)/m,
            lookbehind: true
          },
          'code-language': {
            pattern: /^(```).+/,
            lookbehind: true
          },
          'punctuation': /```/
        }
      }],
      'title': [{
        // title 1
        // =======

        // title 2
        // -------
        pattern: /\S.*(?:\n|\r\n?)(?:==+|--+)(?=[ \t]*$)/m,
        alias: 'important',
        inside: {
          punctuation: /==+$|--+$/
        }
      }, {
        // # title 1
        // ###### title 6
        pattern: /(^\s*)#+.+/m,
        lookbehind: true,
        alias: 'important',
        inside: {
          punctuation: /^#+|#+$/
        }
      }],
      'hr': {
        // ***
        // ---
        // * * *
        // -----------
        pattern: /(^\s*)([*-])(?:[\t ]*\2){2,}(?=\s*$)/m,
        lookbehind: true,
        alias: 'punctuation'
      },
      'list': {
        // * item
        // + item
        // - item
        // 1. item
        pattern: /(^\s*)(?:[*+-]|\d+\.)(?=[\t ].)/m,
        lookbehind: true,
        alias: 'punctuation'
      },
      'url-reference': {
        // [id]: http://example.com "Optional title"
        // [id]: http://example.com 'Optional title'
        // [id]: http://example.com (Optional title)
        // [id]: <http://example.com> "Optional title"
        pattern: /!?\[[^\]]+\]:[\t ]+(?:\S+|<(?:\\.|[^>\\])+>)(?:[\t ]+(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\((?:\\.|[^)\\])*\)))?/,
        inside: {
          'variable': {
            pattern: /^(!?\[)[^\]]+/,
            lookbehind: true
          },
          'string': /(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\((?:\\.|[^)\\])*\))$/,
          'punctuation': /^[\[\]!:]|[<>]/
        },
        alias: 'url'
      },
      'bold': {
        // **strong**
        // __strong__

        // allow one nested instance of italic text using the same delimiter
        pattern: createInline(/__(?:(?!_)<inner>|_(?:(?!_)<inner>)+_)+__/.source, true),
        lookbehind: true,
        greedy: true,
        inside: {
          'content': {
            pattern: /(^..)[\s\S]+(?=..$)/,
            lookbehind: true,
            inside: {} // see below
          },
          'punctuation': /\*\*|__/
        }
      },
      'italic': {
        // *em*
        // _em_

        // allow one nested instance of bold text using the same delimiter
        pattern: createInline(/_(?:(?!_)<inner>|__(?:(?!_)<inner>)+__)+_/.source, true),
        lookbehind: true,
        greedy: true,
        inside: {
          'content': {
            pattern: /(^.)[\s\S]+(?=.$)/,
            lookbehind: true,
            inside: {} // see below
          },
          'punctuation': /[*_]/
        }
      },
      'strike': {
        // ~~strike through~~
        // ~strike~
        pattern: createInline(/(~~?)(?:(?!~)<inner>)+?\2/.source, false),
        lookbehind: true,
        greedy: true,
        inside: {
          'content': {
            pattern: /(^~~?)[\s\S]+(?=\1$)/,
            lookbehind: true,
            inside: {} // see below
          },
          'punctuation': /~~?/
        }
      },
      'url': {
        // [example](http://example.com "Optional title")
        // [example][id]
        // [example] [id]
        pattern: createInline(/!?\[(?:(?!\])<inner>)+\](?:\([^\s)]+(?:[\t ]+"(?:\\.|[^"\\])*")?\)| ?\[(?:(?!\])<inner>)+\])/.source, false),
        lookbehind: true,
        greedy: true,
        inside: {
          'variable': {
            pattern: /(\[)[^\]]+(?=\]$)/,
            lookbehind: true
          },
          'content': {
            pattern: /(^!?\[)[^\]]+(?=\])/,
            lookbehind: true,
            inside: {} // see below
          },
          'string': {
            pattern: /"(?:\\.|[^"\\])*"(?=\)$)/
          }
        }
      }
    });
    ['url', 'bold', 'italic', 'strike'].forEach(function (token) {
      ['url', 'bold', 'italic', 'strike'].forEach(function (inside) {
        if (token !== inside) {
          Prism.languages.markdown[token].inside.content.inside[inside] = Prism.languages.markdown[inside];
        }
      });
    });
    Prism.hooks.add('after-tokenize', function (env) {
      if (env.language !== 'markdown' && env.language !== 'md') {
        return;
      }
      function walkTokens(tokens) {
        if (!tokens || typeof tokens === 'string') {
          return;
        }
        for (var i = 0, l = tokens.length; i < l; i++) {
          var token = tokens[i];
          if (token.type !== 'code') {
            walkTokens(token.content);
            continue;
          }

          /*
           * Add the correct `language-xxxx` class to this code block. Keep in mind that the `code-language` token
           * is optional. But the grammar is defined so that there is only one case we have to handle:
           *
           * token.content = [
           *     <span class="punctuation">```</span>,
           *     <span class="code-language">xxxx</span>,
           *     '\n', // exactly one new lines (\r or \n or \r\n)
           *     <span class="code-block">...</span>,
           *     '\n', // exactly one new lines again
           *     <span class="punctuation">```</span>
           * ];
           */

          var codeLang = token.content[1];
          var codeBlock = token.content[3];
          if (codeLang && codeBlock && codeLang.type === 'code-language' && codeBlock.type === 'code-block' && typeof codeLang.content === 'string') {
            // this might be a language that Prism does not support

            // do some replacements to support C++, C#, and F#
            var lang = codeLang.content.replace(/\b#/g, 'sharp').replace(/\b\+\+/g, 'pp');
            // only use the first word
            lang = (/[a-z][\w-]*/i.exec(lang) || [''])[0].toLowerCase();
            var alias = 'language-' + lang;

            // add alias
            if (!codeBlock.alias) {
              codeBlock.alias = [alias];
            } else if (typeof codeBlock.alias === 'string') {
              codeBlock.alias = [codeBlock.alias, alias];
            } else {
              codeBlock.alias.push(alias);
            }
          }
        }
      }
      walkTokens(env.tokens);
    });
    Prism.hooks.add('wrap', function (env) {
      if (env.type !== 'code-block') {
        return;
      }
      var codeLang = '';
      for (var i = 0, l = env.classes.length; i < l; i++) {
        var cls = env.classes[i];
        var match = /language-(.+)/.exec(cls);
        if (match) {
          codeLang = match[1];
          break;
        }
      }
      var grammar = Prism.languages[codeLang];
      if (!grammar) {
        if (codeLang && codeLang !== 'none' && Prism.plugins.autoloader) {
          var id = 'md-' + new Date().valueOf() + '-' + Math.floor(Math.random() * 1e16);
          env.attributes['id'] = id;
          Prism.plugins.autoloader.loadLanguages(codeLang, function () {
            var ele = document.getElementById(id);
            if (ele) {
              ele.innerHTML = Prism.highlight(ele.textContent, Prism.languages[codeLang], codeLang);
            }
          });
        }
      } else {
        // reverse Prism.util.encode
        var code = env.content.replace(/&lt;/g, '<').replace(/&amp;/g, '&');
        env.content = Prism.highlight(code, grammar, codeLang);
      }
    });
    Prism.languages.md = Prism.languages.markdown;
  })(Prism);
  Prism.languages.julia = {
    'comment': {
      pattern: /(^|[^\\])#.*/,
      lookbehind: true
    },
    'string': /("""|''')[\s\S]+?\1|("|')(?:\\.|(?!\2)[^\\\r\n])*\2/,
    'keyword': /\b(?:abstract|baremodule|begin|bitstype|break|catch|ccall|const|continue|do|else|elseif|end|export|finally|for|function|global|if|immutable|import|importall|in|let|local|macro|module|print|println|quote|return|struct|try|type|typealias|using|while)\b/,
    'boolean': /\b(?:true|false)\b/,
    'number': /(?:\b(?=\d)|\B(?=\.))(?:0[box])?(?:[\da-f]+\.?\d*|\.\d+)(?:[efp][+-]?\d+)?j?/i,
    'operator': /[-+*^%÷&$\\]=?|\/[\/=]?|!=?=?|\|[=>]?|<(?:<=?|[=:])?|>(?:=|>>?=?)?|==?=?|[~≠≤≥]/,
    'punctuation': /[{}[\];(),.:]/,
    'constant': /\b(?:(?:NaN|Inf)(?:16|32|64)?)\b/
  };
  var css = "/**\n * prism.js default theme for JavaScript, CSS and HTML\n * Based on dabblet (http://dabblet.com)\n * @author Lea Verou\n */\n\ncode[class*=\"language-\"],\npre[class*=\"language-\"] {\n\tcolor: black;\n\tbackground: none;\n\ttext-shadow: 0 1px white;\n\tfont-family: Consolas, Monaco, 'Andale Mono', 'Ubuntu Mono', monospace;\n\tfont-size: 1em;\n\ttext-align: left;\n\twhite-space: pre;\n\tword-spacing: normal;\n\tword-break: normal;\n\tword-wrap: normal;\n\tline-height: 1.5;\n\n\t-moz-tab-size: 4;\n\t-o-tab-size: 4;\n\ttab-size: 4;\n\n\t-webkit-hyphens: none;\n\t-moz-hyphens: none;\n\t-ms-hyphens: none;\n\thyphens: none;\n}\n\npre[class*=\"language-\"]::-moz-selection, pre[class*=\"language-\"] ::-moz-selection,\ncode[class*=\"language-\"]::-moz-selection, code[class*=\"language-\"] ::-moz-selection {\n\ttext-shadow: none;\n\tbackground: #b3d4fc;\n}\n\npre[class*=\"language-\"]::selection, pre[class*=\"language-\"] ::selection,\ncode[class*=\"language-\"]::selection, code[class*=\"language-\"] ::selection {\n\ttext-shadow: none;\n\tbackground: #b3d4fc;\n}\n\n@media print {\n\tcode[class*=\"language-\"],\n\tpre[class*=\"language-\"] {\n\t\ttext-shadow: none;\n\t}\n}\n\n/* Code blocks */\npre[class*=\"language-\"] {\n\tpadding: 1em;\n\tmargin: .5em 0;\n\toverflow: auto;\n}\n\n:not(pre) > code[class*=\"language-\"],\npre[class*=\"language-\"] {\n\tbackground: #f5f2f0;\n}\n\n/* Inline code */\n:not(pre) > code[class*=\"language-\"] {\n\tpadding: .1em;\n\tborder-radius: .3em;\n\twhite-space: normal;\n}\n\n.token.comment,\n.token.prolog,\n.token.doctype,\n.token.cdata {\n\tcolor: slategray;\n}\n\n.token.punctuation {\n\tcolor: #999;\n}\n\n.token.namespace {\n\topacity: .7;\n}\n\n.token.property,\n.token.tag,\n.token.boolean,\n.token.number,\n.token.constant,\n.token.symbol,\n.token.deleted {\n\tcolor: #905;\n}\n\n.token.selector,\n.token.attr-name,\n.token.string,\n.token.char,\n.token.builtin,\n.token.inserted {\n\tcolor: #690;\n}\n\n.token.operator,\n.token.entity,\n.token.url,\n.language-css .token.string,\n.style .token.string {\n\tcolor: #9a6e3a;\n\tbackground: hsla(0, 0%, 100%, .5);\n}\n\n.token.atrule,\n.token.attr-value,\n.token.keyword {\n\tcolor: #07a;\n}\n\n.token.function,\n.token.class-name {\n\tcolor: #DD4A68;\n}\n\n.token.regex,\n.token.important,\n.token.variable {\n\tcolor: #e90;\n}\n\n.token.important,\n.token.bold {\n\tfont-weight: bold;\n}\n.token.italic {\n\tfont-style: italic;\n}\n\n.token.entity {\n\tcursor: help;\n}\n";

  // Copyright 2018 The Distill Template Authors

  var T$4 = Template('d-code', "\n<style>\n\ncode {\n  white-space: nowrap;\n  background: rgba(0, 0, 0, 0.04);\n  border-radius: 2px;\n  padding: 4px 7px;\n  font-size: 15px;\n  color: rgba(0, 0, 0, 0.6);\n}\n\npre code {\n  display: block;\n  border-left: 2px solid rgba(0, 0, 0, .1);\n  padding: 0 0 0 36px;\n}\n\n".concat(css, "\n</style>\n\n<code id=\"code-container\"></code>\n\n"));
  var Code = /*#__PURE__*/function (_Mutating2) {
    function Code() {
      _classCallCheck(this, Code);
      return _callSuper(this, Code, arguments);
    }
    _inherits(Code, _Mutating2);
    return _createClass(Code, [{
      key: "renderContent",
      value: function renderContent() {
        // check if language can be highlighted
        this.languageName = this.getAttribute('language');
        if (!this.languageName) {
          console.warn('You need to provide a language attribute to your <d-code> block to let us know how to highlight your code; e.g.:\n <d-code language="python">zeros = np.zeros(shape)</d-code>.');
          return;
        }
        var language = prism.languages[this.languageName];
        if (language == undefined) {
          console.warn("Distill does not yet support highlighting your code block in \"".concat(this.languageName, "'."));
          return;
        }
        var content = this.textContent;
        var codeTag = this.shadowRoot.querySelector('#code-container');
        if (this.hasAttribute('block')) {
          // normalize the tab indents
          content = content.replace(/\n/, '');
          var tabs = content.match(/\s*/);
          content = content.replace(new RegExp('\n' + tabs, 'g'), '\n');
          content = content.trim();
          // wrap code block in pre tag if needed
          if (codeTag.parentNode instanceof ShadowRoot) {
            var preTag = document.createElement('pre');
            this.shadowRoot.removeChild(codeTag);
            preTag.appendChild(codeTag);
            this.shadowRoot.appendChild(preTag);
          }
        }
        codeTag.className = "language-".concat(this.languageName);
        codeTag.innerHTML = prism.highlight(content, language);
      }
    }]);
  }(Mutating(T$4(HTMLElement))); // Copyright 2018 The Distill Template Authors
  var T$5 = Template('d-footnote', "\n<style>\n\nd-math[block] {\n  display: block;\n}\n\n:host {\n\n}\n\nsup {\n  line-height: 1em;\n  font-size: 0.75em;\n  position: relative;\n  top: -.5em;\n  vertical-align: baseline;\n}\n\nspan {\n  color: hsla(206, 90%, 20%, 0.7);\n  cursor: default;\n}\n\n.footnote-container {\n  padding: 10px;\n}\n\n</style>\n\n<d-hover-box>\n  <div class=\"footnote-container\">\n    <slot id=\"slot\"></slot>\n  </div>\n</d-hover-box>\n\n<sup>\n  <span id=\"fn-\" data-hover-ref=\"\"></span>\n</sup>\n\n");
  var Footnote = /*#__PURE__*/function (_T$4) {
    function Footnote() {
      var _this11;
      _classCallCheck(this, Footnote);
      _this11 = _callSuper(this, Footnote);
      var options = {
        childList: true,
        characterData: true,
        subtree: true
      };
      var observer = new MutationObserver(_this11.notify);
      observer.observe(_this11, options);
      return _this11;
    }
    _inherits(Footnote, _T$4);
    return _createClass(Footnote, [{
      key: "notify",
      value: function notify() {
        var options = {
          detail: this,
          bubbles: true
        };
        var event = new CustomEvent('onFootnoteChanged', options);
        document.dispatchEvent(event);
      }
    }, {
      key: "connectedCallback",
      value: function connectedCallback() {
        var _this12 = this;
        // listen and notify about changes to slotted content
        // const slot = this.shadowRoot.querySelector('#slot');
        // console.warn(slot.textContent);
        // slot.addEventListener('slotchange', this.notify);
        this.hoverBox = this.root.querySelector('d-hover-box');
        window.customElements.whenDefined('d-hover-box').then(function () {
          _this12.hoverBox.listen(_this12);
        });
        // create numeric ID
        Footnote.currentFootnoteId += 1;
        var IdString = Footnote.currentFootnoteId.toString();
        this.root.host.id = 'd-footnote-' + IdString;

        // set up hidden hover box
        var id = 'dt-fn-hover-box-' + IdString;
        this.hoverBox.id = id;

        // set up visible footnote marker
        var span = this.root.querySelector('#fn-');
        span.setAttribute('id', 'fn-' + IdString);
        span.setAttribute('data-hover-ref', id);
        span.textContent = IdString;
      }
    }]);
  }(T$5(HTMLElement));
  Footnote.currentFootnoteId = 0;

  // Copyright 2018 The Distill Template Authors

  var T$6 = Template('d-footnote-list', "\n<style>\n\nd-footnote-list {\n  contain: layout style;\n}\n\nd-footnote-list > * {\n  grid-column: text;\n}\n\nd-footnote-list a.footnote-backlink {\n  color: rgba(0,0,0,0.3);\n  padding-left: 0.5em;\n}\n\n</style>\n\n<h3>Footnotes</h3>\n<ol></ol>\n", false);
  var FootnoteList = /*#__PURE__*/function (_T$5) {
    function FootnoteList() {
      _classCallCheck(this, FootnoteList);
      return _callSuper(this, FootnoteList, arguments);
    }
    _inherits(FootnoteList, _T$5);
    return _createClass(FootnoteList, [{
      key: "connectedCallback",
      value: function connectedCallback() {
        _superPropGet(FootnoteList, "connectedCallback", this, 3)([]);
        this.list = this.root.querySelector('ol');
        // footnotes list is initially hidden
        this.root.style.display = 'none';
        // look through document and register existing footnotes
        // Store.subscribeTo('footnotes', (footnote) => {
        //   this.renderFootnote(footnote);
        // });
      }

      // TODO: could optimize this to accept individual footnotes?
    }, {
      key: "footnotes",
      set: function set(footnotes) {
        this.list.innerHTML = '';
        if (footnotes.length) {
          // ensure footnote list is visible
          this.root.style.display = '';
          var _iterator17 = _createForOfIteratorHelper(footnotes),
            _step17;
          try {
            for (_iterator17.s(); !(_step17 = _iterator17.n()).done;) {
              var footnote = _step17.value;
              // construct and append list item to show footnote
              var listItem = document.createElement('li');
              listItem.id = footnote.id + '-listing';
              listItem.innerHTML = footnote.innerHTML;
              var backlink = document.createElement('a');
              backlink.setAttribute('class', 'footnote-backlink');
              backlink.setAttribute('target', '_self');
              backlink.textContent = '[↩]';
              backlink.href = '#' + footnote.id;
              listItem.appendChild(backlink);
              this.list.appendChild(listItem);
            }
          } catch (err) {
            _iterator17.e(err);
          } finally {
            _iterator17.f();
          }
        } else {
          // ensure footnote list is invisible
          this.root.style.display = 'none';
        }
      }
    }]);
  }(T$6(HTMLElement)); // Copyright 2018 The Distill Template Authors
  var T$7 = Template('d-hover-box', "\n<style>\n\n:host {\n  position: absolute;\n  width: 100%;\n  left: 0px;\n  z-index: 10000;\n  display: none;\n  white-space: normal\n}\n\n.container {\n  position: relative;\n  width: 704px;\n  max-width: 100vw;\n  margin: 0 auto;\n}\n\n.panel {\n  position: absolute;\n  font-size: 1rem;\n  line-height: 1.5em;\n  top: 0;\n  left: 0;\n  width: 100%;\n  border: 1px solid rgba(0, 0, 0, 0.1);\n  background-color: rgba(250, 250, 250, 0.95);\n  box-shadow: 0 0 7px rgba(0, 0, 0, 0.1);\n  border-radius: 4px;\n  box-sizing: border-box;\n\n  backdrop-filter: blur(2px);\n  -webkit-backdrop-filter: blur(2px);\n}\n\n</style>\n\n<div class=\"container\">\n  <div class=\"panel\">\n    <slot></slot>\n  </div>\n</div>\n");
  var HoverBox = /*#__PURE__*/function (_T$6) {
    function HoverBox() {
      _classCallCheck(this, HoverBox);
      return _callSuper(this, HoverBox);
    }
    _inherits(HoverBox, _T$6);
    return _createClass(HoverBox, [{
      key: "connectedCallback",
      value: function connectedCallback() {}
    }, {
      key: "listen",
      value: function listen(element) {
        // console.log(element)
        this.bindDivEvents(this);
        this.bindTriggerEvents(element);
        // this.style.display = "block";
      }
    }, {
      key: "bindDivEvents",
      value: function bindDivEvents(element) {
        var _this13 = this;
        // For mice, same behavior as hovering on links
        element.addEventListener('mouseover', function () {
          if (!_this13.visible) _this13.showAtNode(element);
          _this13.stopTimeout();
        });
        element.addEventListener('mouseout', function () {
          _this13.extendTimeout(500);
        });
        // Don't trigger body touchstart event when touching within box
        element.addEventListener('touchstart', function (event) {
          event.stopPropagation();
        }, {
          passive: true
        });
        // Close box when touching outside box
        document.body.addEventListener('touchstart', function () {
          _this13.hide();
        }, {
          passive: true
        });
      }
    }, {
      key: "bindTriggerEvents",
      value: function bindTriggerEvents(node) {
        var _this14 = this;
        node.addEventListener('mouseover', function () {
          if (!_this14.visible) {
            _this14.showAtNode(node);
          }
          _this14.stopTimeout();
        });
        node.addEventListener('mouseout', function () {
          _this14.extendTimeout(300);
        });
        node.addEventListener('touchstart', function (event) {
          if (_this14.visible) {
            _this14.hide();
          } else {
            _this14.showAtNode(node);
          }
          // Don't trigger body touchstart event when touching link
          event.stopPropagation();
        }, {
          passive: true
        });
      }
    }, {
      key: "show",
      value: function show(position) {
        this.visible = true;
        this.style.display = 'block';
        // 10px extra offset from element
        this.style.top = Math.round(position[1] + 10) + 'px';
      }
    }, {
      key: "showAtNode",
      value: function showAtNode(node) {
        // https://developer.mozilla.org/en-US/docs/Web/API/HTMLElement/offsetTop
        var bbox = node.getBoundingClientRect();
        this.show([node.offsetLeft + bbox.width, node.offsetTop + bbox.height]);
      }
    }, {
      key: "hide",
      value: function hide() {
        this.visible = false;
        this.style.display = 'none';
        this.stopTimeout();
      }
    }, {
      key: "stopTimeout",
      value: function stopTimeout() {
        if (this.timeout) {
          clearTimeout(this.timeout);
        }
      }
    }, {
      key: "extendTimeout",
      value: function extendTimeout(time) {
        var _this15 = this;
        this.stopTimeout();
        this.timeout = setTimeout(function () {
          _this15.hide();
        }, time);
      }
    }]);
  }(T$7(HTMLElement)); // Copyright 2018 The Distill Template Authors
  //
  // Licensed under the Apache License, Version 2.0 (the "License");
  // you may not use this file except in compliance with the License.
  // You may obtain a copy of the License at
  //
  //      http://www.apache.org/licenses/LICENSE-2.0
  //
  // Unless required by applicable law or agreed to in writing, software
  // distributed under the License is distributed on an "AS IS" BASIS,
  // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  // See the License for the specific language governing permissions and
  // limitations under the License.
  var Title = /*#__PURE__*/function (_HTMLElement6) {
    function Title() {
      _classCallCheck(this, Title);
      return _callSuper(this, Title, arguments);
    }
    _inherits(Title, _HTMLElement6);
    return _createClass(Title, null, [{
      key: "is",
      get: function get() {
        return 'd-title';
      }
    }]);
  }(/*#__PURE__*/_wrapNativeSuper(HTMLElement)); // Copyright 2018 The Distill Template Authors
  var T$8 = Template('d-references', "\n<style>\nd-references {\n  display: block;\n}\n</style>\n", false);
  var References = /*#__PURE__*/function (_T$7) {
    function References() {
      _classCallCheck(this, References);
      return _callSuper(this, References, arguments);
    }
    _inherits(References, _T$7);
    return _createClass(References);
  }(T$8(HTMLElement)); // Copyright 2018 The Distill Template Authors
  //
  // Licensed under the Apache License, Version 2.0 (the "License");
  // you may not use this file except in compliance with the License.
  // You may obtain a copy of the License at
  //
  //      http://www.apache.org/licenses/LICENSE-2.0
  //
  // Unless required by applicable law or agreed to in writing, software
  // distributed under the License is distributed on an "AS IS" BASIS,
  // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  // See the License for the specific language governing permissions and
  // limitations under the License.
  var TOC = /*#__PURE__*/function (_HTMLElement7) {
    function TOC() {
      _classCallCheck(this, TOC);
      return _callSuper(this, TOC, arguments);
    }
    _inherits(TOC, _HTMLElement7);
    return _createClass(TOC, [{
      key: "connectedCallback",
      value: function connectedCallback() {
        var _this16 = this;
        if (!this.getAttribute('prerendered')) {
          window.onload = function () {
            var article = document.querySelector('d-article');
            var headings = article.querySelectorAll('h2, h3');
            renderTOC(_this16, headings);
          };
        }
      }
    }], [{
      key: "is",
      get: function get() {
        return 'd-toc';
      }
    }]);
  }(/*#__PURE__*/_wrapNativeSuper(HTMLElement));
  function renderTOC(element, headings) {
    var ToC = "\n  <style>\n\n  d-toc {\n    contain: layout style;\n    display: block;\n  }\n\n  d-toc ul {\n    padding-left: 0;\n  }\n\n  d-toc ul > ul {\n    padding-left: 24px;\n  }\n\n  d-toc a {\n    border-bottom: none;\n    text-decoration: none;\n  }\n\n  </style>\n  <nav role=\"navigation\" class=\"table-of-contents\"></nav>\n  <h2>Table of contents</h2>\n  <ul>";
    var _iterator18 = _createForOfIteratorHelper(headings),
      _step18;
    try {
      for (_iterator18.s(); !(_step18 = _iterator18.n()).done;) {
        var el = _step18.value;
        // should element be included in TOC?
        var isInTitle = el.parentElement.tagName == 'D-TITLE';
        var isException = el.getAttribute('no-toc');
        if (isInTitle || isException) continue;
        // create TOC entry
        var _title = el.textContent;
        var link = '#' + el.getAttribute('id');
        var newLine = '<li>' + '<a href="' + link + '">' + _title + '</a>' + '</li>';
        if (el.tagName == 'H3') {
          newLine = '<ul>' + newLine + '</ul>';
        } else {
          newLine += '<br>';
        }
        ToC += newLine;
      }
    } catch (err) {
      _iterator18.e(err);
    } finally {
      _iterator18.f();
    }
    ToC += '</ul></nav>';
    element.innerHTML = ToC;
  }

  // Copyright 2018 The Distill Template Authors
  //
  // Licensed under the Apache License, Version 2.0 (the "License");
  // you may not use this file except in compliance with the License.
  // You may obtain a copy of the License at
  //
  //      http://www.apache.org/licenses/LICENSE-2.0
  //
  // Unless required by applicable law or agreed to in writing, software
  // distributed under the License is distributed on an "AS IS" BASIS,
  // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  // See the License for the specific language governing permissions and
  // limitations under the License.

  // Figure
  //
  // d-figure provides a state-machine of visibility events:
  //
  //                         scroll out of view
  //                         +----------------+
  //   *do work here*        |                |
  // +----------------+    +-+---------+    +-v---------+
  // | ready          +----> onscreen  |    | offscreen |
  // +----------------+    +---------^-+    +---------+-+
  //                                 |                |
  //                                 +----------------+
  //                                  scroll into view
  //
  var Figure = /*#__PURE__*/function (_HTMLElement8) {
    function Figure() {
      var _this17;
      _classCallCheck(this, Figure);
      _this17 = _callSuper(this, Figure);
      // debugger
      _this17._ready = false;
      _this17._onscreen = false;
      _this17._offscreen = true;
      return _this17;
    }
    _inherits(Figure, _HTMLElement8);
    return _createClass(Figure, [{
      key: "connectedCallback",
      value: function connectedCallback() {
        this.loadsWhileScrolling = this.hasAttribute('loadsWhileScrolling');
        Figure.marginObserver.observe(this);
        Figure.directObserver.observe(this);
      }
    }, {
      key: "disconnectedCallback",
      value: function disconnectedCallback() {
        Figure.marginObserver.unobserve(this);
        Figure.directObserver.unobserve(this);
      }

      // We use two separate observers:
      // One with an extra 1000px margin to warn if the viewpoint gets close,
      // And one for the actual on/off screen events
    }, {
      key: "addEventListener",
      value:
      // Notify listeners that registered late, too:

      function addEventListener(eventName, callback) {
        _superPropGet(Figure, "addEventListener", this, 3)([eventName, callback]);
        // if we had already dispatched something while presumingly no one was listening, we do so again
        // debugger
        if (eventName === 'ready') {
          if (Figure.readyQueue.indexOf(this) !== -1) {
            this._ready = false;
            Figure.runReadyQueue();
          }
        }
        if (eventName === 'onscreen') {
          this.onscreen();
        }
      }

      // Custom Events
    }, {
      key: "ready",
      value: function ready() {
        // debugger
        this._ready = true;
        Figure.marginObserver.unobserve(this);
        var event = new CustomEvent('ready');
        this.dispatchEvent(event);
      }
    }, {
      key: "onscreen",
      value: function onscreen() {
        this._onscreen = true;
        this._offscreen = false;
        var event = new CustomEvent('onscreen');
        this.dispatchEvent(event);
      }
    }, {
      key: "offscreen",
      value: function offscreen() {
        this._onscreen = false;
        this._offscreen = true;
        var event = new CustomEvent('offscreen');
        this.dispatchEvent(event);
      }
    }], [{
      key: "is",
      get: function get() {
        return 'd-figure';
      }
    }, {
      key: "readyQueue",
      get: function get() {
        if (!Figure._readyQueue) {
          Figure._readyQueue = [];
        }
        return Figure._readyQueue;
      }
    }, {
      key: "addToReadyQueue",
      value: function addToReadyQueue(figure) {
        if (Figure.readyQueue.indexOf(figure) === -1) {
          Figure.readyQueue.push(figure);
          Figure.runReadyQueue();
        }
      }
    }, {
      key: "runReadyQueue",
      value: function runReadyQueue() {
        // console.log("Checking to run readyQueue, length: " + Figure.readyQueue.length + ", scrolling: " + Figure.isScrolling);
        // if (Figure.isScrolling) return;
        // console.log("Running ready Queue");
        var figure = Figure.readyQueue.sort(function (a, b) {
          return a._seenOnScreen - b._seenOnScreen;
        }).filter(function (figure) {
          return !figure._ready;
        }).pop();
        if (figure) {
          figure.ready();
          requestAnimationFrame(Figure.runReadyQueue);
        }
      }
    }, {
      key: "marginObserver",
      get: function get() {
        if (!Figure._marginObserver) {
          // if (!('IntersectionObserver' in window)) {
          //   throw new Error('no interscetionobbserver!');
          // }
          var viewportHeight = window.innerHeight;
          var margin = Math.floor(2 * viewportHeight);
          var options = {
            rootMargin: margin + 'px 0px ' + margin + 'px 0px',
            threshold: 0.01
          };
          var callback = Figure.didObserveMarginIntersection;
          var observer = new IntersectionObserver(callback, options);
          Figure._marginObserver = observer;
        }
        return Figure._marginObserver;
      }
    }, {
      key: "didObserveMarginIntersection",
      value: function didObserveMarginIntersection(entries) {
        var _iterator19 = _createForOfIteratorHelper(entries),
          _step19;
        try {
          for (_iterator19.s(); !(_step19 = _iterator19.n()).done;) {
            var entry = _step19.value;
            var figure = entry.target;
            if (entry.isIntersecting && !figure._ready) {
              Figure.addToReadyQueue(figure);
            }
          }
        } catch (err) {
          _iterator19.e(err);
        } finally {
          _iterator19.f();
        }
      }
    }, {
      key: "directObserver",
      get: function get() {
        if (!Figure._directObserver) {
          Figure._directObserver = new IntersectionObserver(Figure.didObserveDirectIntersection, {
            rootMargin: '0px',
            threshold: [0, 1.0]
          });
        }
        return Figure._directObserver;
      }
    }, {
      key: "didObserveDirectIntersection",
      value: function didObserveDirectIntersection(entries) {
        var _iterator20 = _createForOfIteratorHelper(entries),
          _step20;
        try {
          for (_iterator20.s(); !(_step20 = _iterator20.n()).done;) {
            var entry = _step20.value;
            var figure = entry.target;
            if (entry.isIntersecting) {
              figure._seenOnScreen = new Date();
              // if (!figure._ready) { figure.ready(); }
              if (figure._offscreen) {
                figure.onscreen();
              }
            } else {
              if (figure._onscreen) {
                figure.offscreen();
              }
            }
          }
        } catch (err) {
          _iterator20.e(err);
        } finally {
          _iterator20.f();
        }
      }
    }]);
  }(/*#__PURE__*/_wrapNativeSuper(HTMLElement));
  if (typeof window !== 'undefined') {
    Figure.isScrolling = false;
    var timeout;
    var resetTimer = function resetTimer() {
      Figure.isScrolling = true;
      clearTimeout(timeout);
      timeout = setTimeout(function () {
        Figure.isScrolling = false;
        Figure.runReadyQueue();
      }, 500);
    };
    window.addEventListener('scroll', resetTimer, true);
  }

  // Copyright 2018 The Distill Template Authors

  // This overlay is not secure.
  // It is only meant as a social deterrent.

  var productionHostname = 'distill.pub';
  var T$9 = Template('d-interstitial', "\n<style>\n\n.overlay {\n  position: fixed;\n  width: 100%;\n  height: 100%;\n  top: 0;\n  left: 0;\n  background: white;\n\n  opacity: 1;\n  visibility: visible;\n\n  display: flex;\n  flex-flow: column;\n  justify-content: center;\n  z-index: 2147483647 /* MaxInt32 */\n\n}\n\n.container {\n  position: relative;\n  margin-left: auto;\n  margin-right: auto;\n  max-width: 420px;\n  padding: 2em;\n}\n\nh1 {\n  text-decoration: underline;\n  text-decoration-color: hsl(0,100%,40%);\n  -webkit-text-decoration-color: hsl(0,100%,40%);\n  margin-bottom: 1em;\n  line-height: 1.5em;\n}\n\ninput[type=\"password\"] {\n  -webkit-appearance: none;\n  -moz-appearance: none;\n  appearance: none;\n  -webkit-box-shadow: none;\n  -moz-box-shadow: none;\n  box-shadow: none;\n  -webkit-border-radius: none;\n  -moz-border-radius: none;\n  -ms-border-radius: none;\n  -o-border-radius: none;\n  border-radius: none;\n  outline: none;\n\n  font-size: 18px;\n  background: none;\n  width: 25%;\n  padding: 10px;\n  border: none;\n  border-bottom: solid 2px #999;\n  transition: border .3s;\n}\n\ninput[type=\"password\"]:focus {\n  border-bottom: solid 2px #333;\n}\n\ninput[type=\"password\"].wrong {\n  border-bottom: solid 2px hsl(0,100%,40%);\n}\n\np small {\n  color: #888;\n}\n\n.logo {\n  position: relative;\n  font-size: 1.5em;\n  margin-bottom: 3em;\n}\n\n.logo svg {\n  width: 36px;\n  position: relative;\n  top: 6px;\n  margin-right: 2px;\n}\n\n.logo svg path {\n  fill: none;\n  stroke: black;\n  stroke-width: 2px;\n}\n\n</style>\n\n<div class=\"overlay\">\n  <div class=\"container\">\n    <h1>This article is in review.</h1>\n    <p>Do not share this URL or the contents of this article. Thank you!</p>\n    <input id=\"interstitial-password-input\" type=\"password\" name=\"password\" autofocus/>\n    <p><small>Enter the password we shared with you as part of the review process to view the article.</small></p>\n  </div>\n</div>\n");
  var Interstitial = /*#__PURE__*/function (_T$8) {
    function Interstitial() {
      _classCallCheck(this, Interstitial);
      return _callSuper(this, Interstitial, arguments);
    }
    _inherits(Interstitial, _T$8);
    return _createClass(Interstitial, [{
      key: "connectedCallback",
      value: function connectedCallback() {
        var _this18 = this;
        if (this.shouldRemoveSelf()) {
          this.parentElement.removeChild(this);
        } else {
          var passwordInput = this.root.querySelector('#interstitial-password-input');
          passwordInput.oninput = function (event) {
            return _this18.passwordChanged(event);
          };
        }
      }
    }, {
      key: "passwordChanged",
      value: function passwordChanged(event) {
        var entered = event.target.value;
        if (entered === this.password) {
          console.log('Correct password entered.');
          this.parentElement.removeChild(this);
          if (typeof Storage !== 'undefined') {
            console.log('Saved that correct password was entered.');
            localStorage.setItem(this.localStorageIdentifier(), 'true');
          }
        }
      }
    }, {
      key: "shouldRemoveSelf",
      value: function shouldRemoveSelf() {
        // should never be visible in production
        if (window && window.location.hostname === productionHostname) {
          console.warn('Interstitial found on production, hiding it.');
          return true;
        }
        // should only have to enter password once
        if (typeof Storage !== 'undefined') {
          if (localStorage.getItem(this.localStorageIdentifier()) === 'true') {
            console.log('Loaded that correct password was entered before; skipping interstitial.');
            return true;
          }
        }
        // otherwise, leave visible
        return false;
      }
    }, {
      key: "localStorageIdentifier",
      value: function localStorageIdentifier() {
        var prefix = 'distill-drafts';
        var suffix = 'interstitial-password-correct';
        return prefix + (window ? window.location.pathname : '-') + suffix;
      }
    }]);
  }(T$9(HTMLElement));
  function ascending(a, b) {
    return a < b ? -1 : a > b ? 1 : a >= b ? 0 : NaN;
  }
  function bisector(compare) {
    if (compare.length === 1) compare = ascendingComparator(compare);
    return {
      left: function left(a, x, lo, hi) {
        if (lo == null) lo = 0;
        if (hi == null) hi = a.length;
        while (lo < hi) {
          var mid = lo + hi >>> 1;
          if (compare(a[mid], x) < 0) lo = mid + 1;else hi = mid;
        }
        return lo;
      },
      right: function right(a, x, lo, hi) {
        if (lo == null) lo = 0;
        if (hi == null) hi = a.length;
        while (lo < hi) {
          var mid = lo + hi >>> 1;
          if (compare(a[mid], x) > 0) hi = mid;else lo = mid + 1;
        }
        return lo;
      }
    };
  }
  function ascendingComparator(f) {
    return function (d, x) {
      return ascending(f(d), x);
    };
  }
  var ascendingBisect = bisector(ascending);
  var bisectRight = ascendingBisect.right;
  function range(start, stop, step) {
    start = +start, stop = +stop, step = (n = arguments.length) < 2 ? (stop = start, start = 0, 1) : n < 3 ? 1 : +step;
    var i = -1,
      n = Math.max(0, Math.ceil((stop - start) / step)) | 0,
      range = new Array(n);
    while (++i < n) {
      range[i] = start + i * step;
    }
    return range;
  }
  var e10 = Math.sqrt(50),
    e5 = Math.sqrt(10),
    e2 = Math.sqrt(2);
  function ticks(start, stop, count) {
    var reverse,
      i = -1,
      n,
      ticks,
      step;
    stop = +stop, start = +start, count = +count;
    if (start === stop && count > 0) return [start];
    if (reverse = stop < start) n = start, start = stop, stop = n;
    if ((step = tickIncrement(start, stop, count)) === 0 || !isFinite(step)) return [];
    if (step > 0) {
      start = Math.ceil(start / step);
      stop = Math.floor(stop / step);
      ticks = new Array(n = Math.ceil(stop - start + 1));
      while (++i < n) ticks[i] = (start + i) * step;
    } else {
      start = Math.floor(start * step);
      stop = Math.ceil(stop * step);
      ticks = new Array(n = Math.ceil(start - stop + 1));
      while (++i < n) ticks[i] = (start - i) / step;
    }
    if (reverse) ticks.reverse();
    return ticks;
  }
  function tickIncrement(start, stop, count) {
    var step = (stop - start) / Math.max(0, count),
      power = Math.floor(Math.log(step) / Math.LN10),
      error = step / Math.pow(10, power);
    return power >= 0 ? (error >= e10 ? 10 : error >= e5 ? 5 : error >= e2 ? 2 : 1) * Math.pow(10, power) : -Math.pow(10, -power) / (error >= e10 ? 10 : error >= e5 ? 5 : error >= e2 ? 2 : 1);
  }
  function tickStep(start, stop, count) {
    var step0 = Math.abs(stop - start) / Math.max(0, count),
      step1 = Math.pow(10, Math.floor(Math.log(step0) / Math.LN10)),
      error = step0 / step1;
    if (error >= e10) step1 *= 10;else if (error >= e5) step1 *= 5;else if (error >= e2) step1 *= 2;
    return stop < start ? -step1 : step1;
  }
  function initRange(domain, range) {
    switch (arguments.length) {
      case 0:
        break;
      case 1:
        this.range(domain);
        break;
      default:
        this.range(range).domain(domain);
        break;
    }
    return this;
  }
  function define(constructor, factory, prototype) {
    constructor.prototype = factory.prototype = prototype;
    prototype.constructor = constructor;
  }
  function extend(parent, definition) {
    var prototype = Object.create(parent.prototype);
    for (var key in definition) prototype[key] = definition[key];
    return prototype;
  }
  function Color() {}
  var _darker = 0.7;
  var _brighter = 1 / _darker;
  var reI = "\\s*([+-]?\\d+)\\s*",
    reN = "\\s*([+-]?\\d*\\.?\\d+(?:[eE][+-]?\\d+)?)\\s*",
    reP = "\\s*([+-]?\\d*\\.?\\d+(?:[eE][+-]?\\d+)?)%\\s*",
    reHex = /^#([0-9a-f]{3,8})$/,
    reRgbInteger = new RegExp("^rgb\\(" + [reI, reI, reI] + "\\)$"),
    reRgbPercent = new RegExp("^rgb\\(" + [reP, reP, reP] + "\\)$"),
    reRgbaInteger = new RegExp("^rgba\\(" + [reI, reI, reI, reN] + "\\)$"),
    reRgbaPercent = new RegExp("^rgba\\(" + [reP, reP, reP, reN] + "\\)$"),
    reHslPercent = new RegExp("^hsl\\(" + [reN, reP, reP] + "\\)$"),
    reHslaPercent = new RegExp("^hsla\\(" + [reN, reP, reP, reN] + "\\)$");
  var named = {
    aliceblue: 0xf0f8ff,
    antiquewhite: 0xfaebd7,
    aqua: 0x00ffff,
    aquamarine: 0x7fffd4,
    azure: 0xf0ffff,
    beige: 0xf5f5dc,
    bisque: 0xffe4c4,
    black: 0x000000,
    blanchedalmond: 0xffebcd,
    blue: 0x0000ff,
    blueviolet: 0x8a2be2,
    brown: 0xa52a2a,
    burlywood: 0xdeb887,
    cadetblue: 0x5f9ea0,
    chartreuse: 0x7fff00,
    chocolate: 0xd2691e,
    coral: 0xff7f50,
    cornflowerblue: 0x6495ed,
    cornsilk: 0xfff8dc,
    crimson: 0xdc143c,
    cyan: 0x00ffff,
    darkblue: 0x00008b,
    darkcyan: 0x008b8b,
    darkgoldenrod: 0xb8860b,
    darkgray: 0xa9a9a9,
    darkgreen: 0x006400,
    darkgrey: 0xa9a9a9,
    darkkhaki: 0xbdb76b,
    darkmagenta: 0x8b008b,
    darkolivegreen: 0x556b2f,
    darkorange: 0xff8c00,
    darkorchid: 0x9932cc,
    darkred: 0x8b0000,
    darksalmon: 0xe9967a,
    darkseagreen: 0x8fbc8f,
    darkslateblue: 0x483d8b,
    darkslategray: 0x2f4f4f,
    darkslategrey: 0x2f4f4f,
    darkturquoise: 0x00ced1,
    darkviolet: 0x9400d3,
    deeppink: 0xff1493,
    deepskyblue: 0x00bfff,
    dimgray: 0x696969,
    dimgrey: 0x696969,
    dodgerblue: 0x1e90ff,
    firebrick: 0xb22222,
    floralwhite: 0xfffaf0,
    forestgreen: 0x228b22,
    fuchsia: 0xff00ff,
    gainsboro: 0xdcdcdc,
    ghostwhite: 0xf8f8ff,
    gold: 0xffd700,
    goldenrod: 0xdaa520,
    gray: 0x808080,
    green: 0x008000,
    greenyellow: 0xadff2f,
    grey: 0x808080,
    honeydew: 0xf0fff0,
    hotpink: 0xff69b4,
    indianred: 0xcd5c5c,
    indigo: 0x4b0082,
    ivory: 0xfffff0,
    khaki: 0xf0e68c,
    lavender: 0xe6e6fa,
    lavenderblush: 0xfff0f5,
    lawngreen: 0x7cfc00,
    lemonchiffon: 0xfffacd,
    lightblue: 0xadd8e6,
    lightcoral: 0xf08080,
    lightcyan: 0xe0ffff,
    lightgoldenrodyellow: 0xfafad2,
    lightgray: 0xd3d3d3,
    lightgreen: 0x90ee90,
    lightgrey: 0xd3d3d3,
    lightpink: 0xffb6c1,
    lightsalmon: 0xffa07a,
    lightseagreen: 0x20b2aa,
    lightskyblue: 0x87cefa,
    lightslategray: 0x778899,
    lightslategrey: 0x778899,
    lightsteelblue: 0xb0c4de,
    lightyellow: 0xffffe0,
    lime: 0x00ff00,
    limegreen: 0x32cd32,
    linen: 0xfaf0e6,
    magenta: 0xff00ff,
    maroon: 0x800000,
    mediumaquamarine: 0x66cdaa,
    mediumblue: 0x0000cd,
    mediumorchid: 0xba55d3,
    mediumpurple: 0x9370db,
    mediumseagreen: 0x3cb371,
    mediumslateblue: 0x7b68ee,
    mediumspringgreen: 0x00fa9a,
    mediumturquoise: 0x48d1cc,
    mediumvioletred: 0xc71585,
    midnightblue: 0x191970,
    mintcream: 0xf5fffa,
    mistyrose: 0xffe4e1,
    moccasin: 0xffe4b5,
    navajowhite: 0xffdead,
    navy: 0x000080,
    oldlace: 0xfdf5e6,
    olive: 0x808000,
    olivedrab: 0x6b8e23,
    orange: 0xffa500,
    orangered: 0xff4500,
    orchid: 0xda70d6,
    palegoldenrod: 0xeee8aa,
    palegreen: 0x98fb98,
    paleturquoise: 0xafeeee,
    palevioletred: 0xdb7093,
    papayawhip: 0xffefd5,
    peachpuff: 0xffdab9,
    peru: 0xcd853f,
    pink: 0xffc0cb,
    plum: 0xdda0dd,
    powderblue: 0xb0e0e6,
    purple: 0x800080,
    rebeccapurple: 0x663399,
    red: 0xff0000,
    rosybrown: 0xbc8f8f,
    royalblue: 0x4169e1,
    saddlebrown: 0x8b4513,
    salmon: 0xfa8072,
    sandybrown: 0xf4a460,
    seagreen: 0x2e8b57,
    seashell: 0xfff5ee,
    sienna: 0xa0522d,
    silver: 0xc0c0c0,
    skyblue: 0x87ceeb,
    slateblue: 0x6a5acd,
    slategray: 0x708090,
    slategrey: 0x708090,
    snow: 0xfffafa,
    springgreen: 0x00ff7f,
    steelblue: 0x4682b4,
    tan: 0xd2b48c,
    teal: 0x008080,
    thistle: 0xd8bfd8,
    tomato: 0xff6347,
    turquoise: 0x40e0d0,
    violet: 0xee82ee,
    wheat: 0xf5deb3,
    white: 0xffffff,
    whitesmoke: 0xf5f5f5,
    yellow: 0xffff00,
    yellowgreen: 0x9acd32
  };
  define(Color, color, {
    copy: function copy(channels) {
      return Object.assign(new this.constructor(), this, channels);
    },
    displayable: function displayable() {
      return this.rgb().displayable();
    },
    hex: color_formatHex,
    // Deprecated! Use color.formatHex.
    formatHex: color_formatHex,
    formatHsl: color_formatHsl,
    formatRgb: color_formatRgb,
    toString: color_formatRgb
  });
  function color_formatHex() {
    return this.rgb().formatHex();
  }
  function color_formatHsl() {
    return hslConvert(this).formatHsl();
  }
  function color_formatRgb() {
    return this.rgb().formatRgb();
  }
  function color(format) {
    var m, l;
    format = (format + "").trim().toLowerCase();
    return (m = reHex.exec(format)) ? (l = m[1].length, m = parseInt(m[1], 16), l === 6 ? rgbn(m) // #ff0000
    : l === 3 ? new Rgb(m >> 8 & 0xf | m >> 4 & 0xf0, m >> 4 & 0xf | m & 0xf0, (m & 0xf) << 4 | m & 0xf, 1) // #f00
    : l === 8 ? rgba(m >> 24 & 0xff, m >> 16 & 0xff, m >> 8 & 0xff, (m & 0xff) / 0xff) // #ff000000
    : l === 4 ? rgba(m >> 12 & 0xf | m >> 8 & 0xf0, m >> 8 & 0xf | m >> 4 & 0xf0, m >> 4 & 0xf | m & 0xf0, ((m & 0xf) << 4 | m & 0xf) / 0xff) // #f000
    : null // invalid hex
    ) : (m = reRgbInteger.exec(format)) ? new Rgb(m[1], m[2], m[3], 1) // rgb(255, 0, 0)
    : (m = reRgbPercent.exec(format)) ? new Rgb(m[1] * 255 / 100, m[2] * 255 / 100, m[3] * 255 / 100, 1) // rgb(100%, 0%, 0%)
    : (m = reRgbaInteger.exec(format)) ? rgba(m[1], m[2], m[3], m[4]) // rgba(255, 0, 0, 1)
    : (m = reRgbaPercent.exec(format)) ? rgba(m[1] * 255 / 100, m[2] * 255 / 100, m[3] * 255 / 100, m[4]) // rgb(100%, 0%, 0%, 1)
    : (m = reHslPercent.exec(format)) ? hsla(m[1], m[2] / 100, m[3] / 100, 1) // hsl(120, 50%, 50%)
    : (m = reHslaPercent.exec(format)) ? hsla(m[1], m[2] / 100, m[3] / 100, m[4]) // hsla(120, 50%, 50%, 1)
    : named.hasOwnProperty(format) ? rgbn(named[format]) // eslint-disable-line no-prototype-builtins
    : format === "transparent" ? new Rgb(NaN, NaN, NaN, 0) : null;
  }
  function rgbn(n) {
    return new Rgb(n >> 16 & 0xff, n >> 8 & 0xff, n & 0xff, 1);
  }
  function rgba(r, g, b, a) {
    if (a <= 0) r = g = b = NaN;
    return new Rgb(r, g, b, a);
  }
  function rgbConvert(o) {
    if (!(o instanceof Color)) o = color(o);
    if (!o) return new Rgb();
    o = o.rgb();
    return new Rgb(o.r, o.g, o.b, o.opacity);
  }
  function rgb(r, g, b, opacity) {
    return arguments.length === 1 ? rgbConvert(r) : new Rgb(r, g, b, opacity == null ? 1 : opacity);
  }
  function Rgb(r, g, b, opacity) {
    this.r = +r;
    this.g = +g;
    this.b = +b;
    this.opacity = +opacity;
  }
  define(Rgb, rgb, extend(Color, {
    brighter: function brighter(k) {
      k = k == null ? _brighter : Math.pow(_brighter, k);
      return new Rgb(this.r * k, this.g * k, this.b * k, this.opacity);
    },
    darker: function darker(k) {
      k = k == null ? _darker : Math.pow(_darker, k);
      return new Rgb(this.r * k, this.g * k, this.b * k, this.opacity);
    },
    rgb: function rgb() {
      return this;
    },
    displayable: function displayable() {
      return -0.5 <= this.r && this.r < 255.5 && -0.5 <= this.g && this.g < 255.5 && -0.5 <= this.b && this.b < 255.5 && 0 <= this.opacity && this.opacity <= 1;
    },
    hex: rgb_formatHex,
    // Deprecated! Use color.formatHex.
    formatHex: rgb_formatHex,
    formatRgb: rgb_formatRgb,
    toString: rgb_formatRgb
  }));
  function rgb_formatHex() {
    return "#" + hex(this.r) + hex(this.g) + hex(this.b);
  }
  function rgb_formatRgb() {
    var a = this.opacity;
    a = isNaN(a) ? 1 : Math.max(0, Math.min(1, a));
    return (a === 1 ? "rgb(" : "rgba(") + Math.max(0, Math.min(255, Math.round(this.r) || 0)) + ", " + Math.max(0, Math.min(255, Math.round(this.g) || 0)) + ", " + Math.max(0, Math.min(255, Math.round(this.b) || 0)) + (a === 1 ? ")" : ", " + a + ")");
  }
  function hex(value) {
    value = Math.max(0, Math.min(255, Math.round(value) || 0));
    return (value < 16 ? "0" : "") + value.toString(16);
  }
  function hsla(h, s, l, a) {
    if (a <= 0) h = s = l = NaN;else if (l <= 0 || l >= 1) h = s = NaN;else if (s <= 0) h = NaN;
    return new Hsl(h, s, l, a);
  }
  function hslConvert(o) {
    if (o instanceof Hsl) return new Hsl(o.h, o.s, o.l, o.opacity);
    if (!(o instanceof Color)) o = color(o);
    if (!o) return new Hsl();
    if (o instanceof Hsl) return o;
    o = o.rgb();
    var r = o.r / 255,
      g = o.g / 255,
      b = o.b / 255,
      min = Math.min(r, g, b),
      max = Math.max(r, g, b),
      h = NaN,
      s = max - min,
      l = (max + min) / 2;
    if (s) {
      if (r === max) h = (g - b) / s + (g < b) * 6;else if (g === max) h = (b - r) / s + 2;else h = (r - g) / s + 4;
      s /= l < 0.5 ? max + min : 2 - max - min;
      h *= 60;
    } else {
      s = l > 0 && l < 1 ? 0 : h;
    }
    return new Hsl(h, s, l, o.opacity);
  }
  function hsl(h, s, l, opacity) {
    return arguments.length === 1 ? hslConvert(h) : new Hsl(h, s, l, opacity == null ? 1 : opacity);
  }
  function Hsl(h, s, l, opacity) {
    this.h = +h;
    this.s = +s;
    this.l = +l;
    this.opacity = +opacity;
  }
  define(Hsl, hsl, extend(Color, {
    brighter: function brighter(k) {
      k = k == null ? _brighter : Math.pow(_brighter, k);
      return new Hsl(this.h, this.s, this.l * k, this.opacity);
    },
    darker: function darker(k) {
      k = k == null ? _darker : Math.pow(_darker, k);
      return new Hsl(this.h, this.s, this.l * k, this.opacity);
    },
    rgb: function rgb() {
      var h = this.h % 360 + (this.h < 0) * 360,
        s = isNaN(h) || isNaN(this.s) ? 0 : this.s,
        l = this.l,
        m2 = l + (l < 0.5 ? l : 1 - l) * s,
        m1 = 2 * l - m2;
      return new Rgb(hsl2rgb(h >= 240 ? h - 240 : h + 120, m1, m2), hsl2rgb(h, m1, m2), hsl2rgb(h < 120 ? h + 240 : h - 120, m1, m2), this.opacity);
    },
    displayable: function displayable() {
      return (0 <= this.s && this.s <= 1 || isNaN(this.s)) && 0 <= this.l && this.l <= 1 && 0 <= this.opacity && this.opacity <= 1;
    },
    formatHsl: function formatHsl() {
      var a = this.opacity;
      a = isNaN(a) ? 1 : Math.max(0, Math.min(1, a));
      return (a === 1 ? "hsl(" : "hsla(") + (this.h || 0) + ", " + (this.s || 0) * 100 + "%, " + (this.l || 0) * 100 + "%" + (a === 1 ? ")" : ", " + a + ")");
    }
  }));

  /* From FvD 13.37, CSS Color Module Level 3 */
  function hsl2rgb(h, m1, m2) {
    return (h < 60 ? m1 + (m2 - m1) * h / 60 : h < 180 ? m2 : h < 240 ? m1 + (m2 - m1) * (240 - h) / 60 : m1) * 255;
  }
  var deg2rad = Math.PI / 180;
  var rad2deg = 180 / Math.PI;

  // https://observablehq.com/@mbostock/lab-and-rgb
  var K = 18,
    Xn = 0.96422,
    Yn = 1,
    Zn = 0.82521,
    t0 = 4 / 29,
    t1 = 6 / 29,
    t2 = 3 * t1 * t1,
    t3 = t1 * t1 * t1;
  function labConvert(o) {
    if (o instanceof Lab) return new Lab(o.l, o.a, o.b, o.opacity);
    if (o instanceof Hcl) return hcl2lab(o);
    if (!(o instanceof Rgb)) o = rgbConvert(o);
    var r = rgb2lrgb(o.r),
      g = rgb2lrgb(o.g),
      b = rgb2lrgb(o.b),
      y = xyz2lab((0.2225045 * r + 0.7168786 * g + 0.0606169 * b) / Yn),
      x,
      z;
    if (r === g && g === b) x = z = y;else {
      x = xyz2lab((0.4360747 * r + 0.3850649 * g + 0.1430804 * b) / Xn);
      z = xyz2lab((0.0139322 * r + 0.0971045 * g + 0.7141733 * b) / Zn);
    }
    return new Lab(116 * y - 16, 500 * (x - y), 200 * (y - z), o.opacity);
  }
  function lab(l, a, b, opacity) {
    return arguments.length === 1 ? labConvert(l) : new Lab(l, a, b, opacity == null ? 1 : opacity);
  }
  function Lab(l, a, b, opacity) {
    this.l = +l;
    this.a = +a;
    this.b = +b;
    this.opacity = +opacity;
  }
  define(Lab, lab, extend(Color, {
    brighter: function brighter(k) {
      return new Lab(this.l + K * (k == null ? 1 : k), this.a, this.b, this.opacity);
    },
    darker: function darker(k) {
      return new Lab(this.l - K * (k == null ? 1 : k), this.a, this.b, this.opacity);
    },
    rgb: function rgb() {
      var y = (this.l + 16) / 116,
        x = isNaN(this.a) ? y : y + this.a / 500,
        z = isNaN(this.b) ? y : y - this.b / 200;
      x = Xn * lab2xyz(x);
      y = Yn * lab2xyz(y);
      z = Zn * lab2xyz(z);
      return new Rgb(lrgb2rgb(3.1338561 * x - 1.6168667 * y - 0.4906146 * z), lrgb2rgb(-0.9787684 * x + 1.9161415 * y + 0.0334540 * z), lrgb2rgb(0.0719453 * x - 0.2289914 * y + 1.4052427 * z), this.opacity);
    }
  }));
  function xyz2lab(t) {
    return t > t3 ? Math.pow(t, 1 / 3) : t / t2 + t0;
  }
  function lab2xyz(t) {
    return t > t1 ? t * t * t : t2 * (t - t0);
  }
  function lrgb2rgb(x) {
    return 255 * (x <= 0.0031308 ? 12.92 * x : 1.055 * Math.pow(x, 1 / 2.4) - 0.055);
  }
  function rgb2lrgb(x) {
    return (x /= 255) <= 0.04045 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4);
  }
  function hclConvert(o) {
    if (o instanceof Hcl) return new Hcl(o.h, o.c, o.l, o.opacity);
    if (!(o instanceof Lab)) o = labConvert(o);
    if (o.a === 0 && o.b === 0) return new Hcl(NaN, 0 < o.l && o.l < 100 ? 0 : NaN, o.l, o.opacity);
    var h = Math.atan2(o.b, o.a) * rad2deg;
    return new Hcl(h < 0 ? h + 360 : h, Math.sqrt(o.a * o.a + o.b * o.b), o.l, o.opacity);
  }
  function hcl(h, c, l, opacity) {
    return arguments.length === 1 ? hclConvert(h) : new Hcl(h, c, l, opacity == null ? 1 : opacity);
  }
  function Hcl(h, c, l, opacity) {
    this.h = +h;
    this.c = +c;
    this.l = +l;
    this.opacity = +opacity;
  }
  function hcl2lab(o) {
    if (isNaN(o.h)) return new Lab(o.l, 0, 0, o.opacity);
    var h = o.h * deg2rad;
    return new Lab(o.l, Math.cos(h) * o.c, Math.sin(h) * o.c, o.opacity);
  }
  define(Hcl, hcl, extend(Color, {
    brighter: function brighter(k) {
      return new Hcl(this.h, this.c, this.l + K * (k == null ? 1 : k), this.opacity);
    },
    darker: function darker(k) {
      return new Hcl(this.h, this.c, this.l - K * (k == null ? 1 : k), this.opacity);
    },
    rgb: function rgb() {
      return hcl2lab(this).rgb();
    }
  }));
  var A = -0.14861,
    B = +1.78277,
    C = -0.29227,
    D = -0.90649,
    E = +1.97294,
    ED = E * D,
    EB = E * B,
    BC_DA = B * C - D * A;
  function cubehelixConvert(o) {
    if (o instanceof Cubehelix) return new Cubehelix(o.h, o.s, o.l, o.opacity);
    if (!(o instanceof Rgb)) o = rgbConvert(o);
    var r = o.r / 255,
      g = o.g / 255,
      b = o.b / 255,
      l = (BC_DA * b + ED * r - EB * g) / (BC_DA + ED - EB),
      bl = b - l,
      k = (E * (g - l) - C * bl) / D,
      s = Math.sqrt(k * k + bl * bl) / (E * l * (1 - l)),
      // NaN if l=0 or l=1
      h = s ? Math.atan2(k, bl) * rad2deg - 120 : NaN;
    return new Cubehelix(h < 0 ? h + 360 : h, s, l, o.opacity);
  }
  function cubehelix(h, s, l, opacity) {
    return arguments.length === 1 ? cubehelixConvert(h) : new Cubehelix(h, s, l, opacity == null ? 1 : opacity);
  }
  function Cubehelix(h, s, l, opacity) {
    this.h = +h;
    this.s = +s;
    this.l = +l;
    this.opacity = +opacity;
  }
  define(Cubehelix, cubehelix, extend(Color, {
    brighter: function brighter(k) {
      k = k == null ? _brighter : Math.pow(_brighter, k);
      return new Cubehelix(this.h, this.s, this.l * k, this.opacity);
    },
    darker: function darker(k) {
      k = k == null ? _darker : Math.pow(_darker, k);
      return new Cubehelix(this.h, this.s, this.l * k, this.opacity);
    },
    rgb: function rgb() {
      var h = isNaN(this.h) ? 0 : (this.h + 120) * deg2rad,
        l = +this.l,
        a = isNaN(this.s) ? 0 : this.s * l * (1 - l),
        cosh = Math.cos(h),
        sinh = Math.sin(h);
      return new Rgb(255 * (l + a * (A * cosh + B * sinh)), 255 * (l + a * (C * cosh + D * sinh)), 255 * (l + a * (E * cosh)), this.opacity);
    }
  }));
  function constant(x) {
    return function () {
      return x;
    };
  }
  function linear(a, d) {
    return function (t) {
      return a + t * d;
    };
  }
  function exponential(a, b, y) {
    return a = Math.pow(a, y), b = Math.pow(b, y) - a, y = 1 / y, function (t) {
      return Math.pow(a + t * b, y);
    };
  }
  function gamma(y) {
    return (y = +y) === 1 ? nogamma : function (a, b) {
      return b - a ? exponential(a, b, y) : constant(isNaN(a) ? b : a);
    };
  }
  function nogamma(a, b) {
    var d = b - a;
    return d ? linear(a, d) : constant(isNaN(a) ? b : a);
  }
  var rgb$1 = function rgbGamma(y) {
    var color = gamma(y);
    function rgb$1(start, end) {
      var r = color((start = rgb(start)).r, (end = rgb(end)).r),
        g = color(start.g, end.g),
        b = color(start.b, end.b),
        opacity = nogamma(start.opacity, end.opacity);
      return function (t) {
        start.r = r(t);
        start.g = g(t);
        start.b = b(t);
        start.opacity = opacity(t);
        return start + "";
      };
    }
    rgb$1.gamma = rgbGamma;
    return rgb$1;
  }(1);
  function numberArray(a, b) {
    if (!b) b = [];
    var n = a ? Math.min(b.length, a.length) : 0,
      c = b.slice(),
      i;
    return function (t) {
      for (i = 0; i < n; ++i) c[i] = a[i] * (1 - t) + b[i] * t;
      return c;
    };
  }
  function isNumberArray(x) {
    return ArrayBuffer.isView(x) && !(x instanceof DataView);
  }
  function genericArray(a, b) {
    var nb = b ? b.length : 0,
      na = a ? Math.min(nb, a.length) : 0,
      x = new Array(na),
      c = new Array(nb),
      i;
    for (i = 0; i < na; ++i) x[i] = interpolate(a[i], b[i]);
    for (; i < nb; ++i) c[i] = b[i];
    return function (t) {
      for (i = 0; i < na; ++i) c[i] = x[i](t);
      return c;
    };
  }
  function date(a, b) {
    var d = new Date();
    return a = +a, b = +b, function (t) {
      return d.setTime(a * (1 - t) + b * t), d;
    };
  }
  function interpolateNumber(a, b) {
    return a = +a, b = +b, function (t) {
      return a * (1 - t) + b * t;
    };
  }
  function object(a, b) {
    var i = {},
      c = {},
      k;
    if (a === null || _typeof(a) !== "object") a = {};
    if (b === null || _typeof(b) !== "object") b = {};
    for (k in b) {
      if (k in a) {
        i[k] = interpolate(a[k], b[k]);
      } else {
        c[k] = b[k];
      }
    }
    return function (t) {
      for (k in i) c[k] = i[k](t);
      return c;
    };
  }
  var reA = /[-+]?(?:\d+\.?\d*|\.?\d+)(?:[eE][-+]?\d+)?/g,
    reB = new RegExp(reA.source, "g");
  function zero(b) {
    return function () {
      return b;
    };
  }
  function one(b) {
    return function (t) {
      return b(t) + "";
    };
  }
  function string(a, b) {
    var bi = reA.lastIndex = reB.lastIndex = 0,
      // scan index for next number in b
      am,
      // current match in a
      bm,
      // current match in b
      bs,
      // string preceding current number in b, if any
      i = -1,
      // index in s
      s = [],
      // string constants and placeholders
      q = []; // number interpolators

    // Coerce inputs to strings.
    a = a + "", b = b + "";

    // Interpolate pairs of numbers in a & b.
    while ((am = reA.exec(a)) && (bm = reB.exec(b))) {
      if ((bs = bm.index) > bi) {
        // a string precedes the next number in b
        bs = b.slice(bi, bs);
        if (s[i]) s[i] += bs; // coalesce with previous string
        else s[++i] = bs;
      }
      if ((am = am[0]) === (bm = bm[0])) {
        // numbers in a & b match
        if (s[i]) s[i] += bm; // coalesce with previous string
        else s[++i] = bm;
      } else {
        // interpolate non-matching numbers
        s[++i] = null;
        q.push({
          i: i,
          x: interpolateNumber(am, bm)
        });
      }
      bi = reB.lastIndex;
    }

    // Add remains of b.
    if (bi < b.length) {
      bs = b.slice(bi);
      if (s[i]) s[i] += bs; // coalesce with previous string
      else s[++i] = bs;
    }

    // Special optimization for only a single match.
    // Otherwise, interpolate each of the numbers and rejoin the string.
    return s.length < 2 ? q[0] ? one(q[0].x) : zero(b) : (b = q.length, function (t) {
      for (var i = 0, o; i < b; ++i) s[(o = q[i]).i] = o.x(t);
      return s.join("");
    });
  }
  function interpolate(a, b) {
    var t = _typeof(b),
      c;
    return b == null || t === "boolean" ? constant(b) : (t === "number" ? interpolateNumber : t === "string" ? (c = color(b)) ? (b = c, rgb$1) : string : b instanceof color ? rgb$1 : b instanceof Date ? date : isNumberArray(b) ? numberArray : Array.isArray(b) ? genericArray : typeof b.valueOf !== "function" && typeof b.toString !== "function" || isNaN(b) ? object : interpolateNumber)(a, b);
  }
  function interpolateRound(a, b) {
    return a = +a, b = +b, function (t) {
      return Math.round(a * (1 - t) + b * t);
    };
  }
  function constant$1(x) {
    return function () {
      return x;
    };
  }
  function number(x) {
    return +x;
  }
  var unit = [0, 1];
  function identity(x) {
    return x;
  }
  function normalize(a, b) {
    return (b -= a = +a) ? function (x) {
      return (x - a) / b;
    } : constant$1(isNaN(b) ? NaN : 0.5);
  }
  function clamper(a, b) {
    var t;
    if (a > b) t = a, a = b, b = t;
    return function (x) {
      return Math.max(a, Math.min(b, x));
    };
  }

  // normalize(a, b)(x) takes a domain value x in [a,b] and returns the corresponding parameter t in [0,1].
  // interpolate(a, b)(t) takes a parameter t in [0,1] and returns the corresponding range value x in [a,b].
  function bimap(domain, range, interpolate) {
    var d0 = domain[0],
      d1 = domain[1],
      r0 = range[0],
      r1 = range[1];
    if (d1 < d0) d0 = normalize(d1, d0), r0 = interpolate(r1, r0);else d0 = normalize(d0, d1), r0 = interpolate(r0, r1);
    return function (x) {
      return r0(d0(x));
    };
  }
  function polymap(domain, range, interpolate) {
    var j = Math.min(domain.length, range.length) - 1,
      d = new Array(j),
      r = new Array(j),
      i = -1;

    // Reverse descending domains.
    if (domain[j] < domain[0]) {
      domain = domain.slice().reverse();
      range = range.slice().reverse();
    }
    while (++i < j) {
      d[i] = normalize(domain[i], domain[i + 1]);
      r[i] = interpolate(range[i], range[i + 1]);
    }
    return function (x) {
      var i = bisectRight(domain, x, 1, j) - 1;
      return r[i](d[i](x));
    };
  }
  function copy(source, target) {
    return target.domain(source.domain()).range(source.range()).interpolate(source.interpolate()).clamp(source.clamp()).unknown(source.unknown());
  }
  function transformer() {
    var domain = unit,
      range = unit,
      interpolate$1 = interpolate,
      transform,
      untransform,
      unknown,
      clamp = identity,
      piecewise,
      output,
      input;
    function rescale() {
      var n = Math.min(domain.length, range.length);
      if (clamp !== identity) clamp = clamper(domain[0], domain[n - 1]);
      piecewise = n > 2 ? polymap : bimap;
      output = input = null;
      return scale;
    }
    function scale(x) {
      return isNaN(x = +x) ? unknown : (output || (output = piecewise(domain.map(transform), range, interpolate$1)))(transform(clamp(x)));
    }
    scale.invert = function (y) {
      return clamp(untransform((input || (input = piecewise(range, domain.map(transform), interpolateNumber)))(y)));
    };
    scale.domain = function (_) {
      return arguments.length ? (domain = Array.from(_, number), rescale()) : domain.slice();
    };
    scale.range = function (_) {
      return arguments.length ? (range = Array.from(_), rescale()) : range.slice();
    };
    scale.rangeRound = function (_) {
      return range = Array.from(_), interpolate$1 = interpolateRound, rescale();
    };
    scale.clamp = function (_) {
      return arguments.length ? (clamp = _ ? true : identity, rescale()) : clamp !== identity;
    };
    scale.interpolate = function (_) {
      return arguments.length ? (interpolate$1 = _, rescale()) : interpolate$1;
    };
    scale.unknown = function (_) {
      return arguments.length ? (unknown = _, scale) : unknown;
    };
    return function (t, u) {
      transform = t, untransform = u;
      return rescale();
    };
  }
  function continuous() {
    return transformer()(identity, identity);
  }

  // Computes the decimal coefficient and exponent of the specified number x with
  // significant digits p, where x is positive and p is in [1, 21] or undefined.
  // For example, formatDecimal(1.23) returns ["123", 0].
  function formatDecimal(x, p) {
    if ((i = (x = p ? x.toExponential(p - 1) : x.toExponential()).indexOf("e")) < 0) return null; // NaN, ±Infinity
    var i,
      coefficient = x.slice(0, i);

    // The string returned by toExponential either has the form \d\.\d+e[-+]\d+
    // (e.g., 1.2e+3) or the form \de[-+]\d+ (e.g., 1e+3).
    return [coefficient.length > 1 ? coefficient[0] + coefficient.slice(2) : coefficient, +x.slice(i + 1)];
  }
  function exponent(x) {
    return x = formatDecimal(Math.abs(x)), x ? x[1] : NaN;
  }
  function formatGroup(grouping, thousands) {
    return function (value, width) {
      var i = value.length,
        t = [],
        j = 0,
        g = grouping[0],
        length = 0;
      while (i > 0 && g > 0) {
        if (length + g + 1 > width) g = Math.max(1, width - length);
        t.push(value.substring(i -= g, i + g));
        if ((length += g + 1) > width) break;
        g = grouping[j = (j + 1) % grouping.length];
      }
      return t.reverse().join(thousands);
    };
  }
  function formatNumerals(numerals) {
    return function (value) {
      return value.replace(/[0-9]/g, function (i) {
        return numerals[+i];
      });
    };
  }

  // [[fill]align][sign][symbol][0][width][,][.precision][~][type]
  var re = /^(?:(.)?([<>=^]))?([+\-( ])?([$#])?(0)?(\d+)?(,)?(\.\d+)?(~)?([a-z%])?$/i;
  function formatSpecifier(specifier) {
    if (!(match = re.exec(specifier))) throw new Error("invalid format: " + specifier);
    var match;
    return new FormatSpecifier({
      fill: match[1],
      align: match[2],
      sign: match[3],
      symbol: match[4],
      zero: match[5],
      width: match[6],
      comma: match[7],
      precision: match[8] && match[8].slice(1),
      trim: match[9],
      type: match[10]
    });
  }
  formatSpecifier.prototype = FormatSpecifier.prototype; // instanceof

  function FormatSpecifier(specifier) {
    this.fill = specifier.fill === undefined ? " " : specifier.fill + "";
    this.align = specifier.align === undefined ? ">" : specifier.align + "";
    this.sign = specifier.sign === undefined ? "-" : specifier.sign + "";
    this.symbol = specifier.symbol === undefined ? "" : specifier.symbol + "";
    this.zero = !!specifier.zero;
    this.width = specifier.width === undefined ? undefined : +specifier.width;
    this.comma = !!specifier.comma;
    this.precision = specifier.precision === undefined ? undefined : +specifier.precision;
    this.trim = !!specifier.trim;
    this.type = specifier.type === undefined ? "" : specifier.type + "";
  }
  FormatSpecifier.prototype.toString = function () {
    return this.fill + this.align + this.sign + this.symbol + (this.zero ? "0" : "") + (this.width === undefined ? "" : Math.max(1, this.width | 0)) + (this.comma ? "," : "") + (this.precision === undefined ? "" : "." + Math.max(0, this.precision | 0)) + (this.trim ? "~" : "") + this.type;
  };

  // Trims insignificant zeros, e.g., replaces 1.2000k with 1.2k.
  function formatTrim(s) {
    out: for (var n = s.length, i = 1, i0 = -1, i1; i < n; ++i) {
      switch (s[i]) {
        case ".":
          i0 = i1 = i;
          break;
        case "0":
          if (i0 === 0) i0 = i;
          i1 = i;
          break;
        default:
          if (!+s[i]) break out;
          if (i0 > 0) i0 = 0;
          break;
      }
    }
    return i0 > 0 ? s.slice(0, i0) + s.slice(i1 + 1) : s;
  }
  var prefixExponent;
  function formatPrefixAuto(x, p) {
    var d = formatDecimal(x, p);
    if (!d) return x + "";
    var coefficient = d[0],
      exponent = d[1],
      i = exponent - (prefixExponent = Math.max(-8, Math.min(8, Math.floor(exponent / 3))) * 3) + 1,
      n = coefficient.length;
    return i === n ? coefficient : i > n ? coefficient + new Array(i - n + 1).join("0") : i > 0 ? coefficient.slice(0, i) + "." + coefficient.slice(i) : "0." + new Array(1 - i).join("0") + formatDecimal(x, Math.max(0, p + i - 1))[0]; // less than 1y!
  }
  function formatRounded(x, p) {
    var d = formatDecimal(x, p);
    if (!d) return x + "";
    var coefficient = d[0],
      exponent = d[1];
    return exponent < 0 ? "0." + new Array(-exponent).join("0") + coefficient : coefficient.length > exponent + 1 ? coefficient.slice(0, exponent + 1) + "." + coefficient.slice(exponent + 1) : coefficient + new Array(exponent - coefficient.length + 2).join("0");
  }
  var formatTypes = {
    "%": function _(x, p) {
      return (x * 100).toFixed(p);
    },
    "b": function b(x) {
      return Math.round(x).toString(2);
    },
    "c": function c(x) {
      return x + "";
    },
    "d": function d(x) {
      return Math.round(x).toString(10);
    },
    "e": function e(x, p) {
      return x.toExponential(p);
    },
    "f": function f(x, p) {
      return x.toFixed(p);
    },
    "g": function g(x, p) {
      return x.toPrecision(p);
    },
    "o": function o(x) {
      return Math.round(x).toString(8);
    },
    "p": function p(x, _p) {
      return formatRounded(x * 100, _p);
    },
    "r": formatRounded,
    "s": formatPrefixAuto,
    "X": function X(x) {
      return Math.round(x).toString(16).toUpperCase();
    },
    "x": function x(_x) {
      return Math.round(_x).toString(16);
    }
  };
  function identity$1(x) {
    return x;
  }
  var map = Array.prototype.map,
    prefixes = ["y", "z", "a", "f", "p", "n", "µ", "m", "", "k", "M", "G", "T", "P", "E", "Z", "Y"];
  function formatLocale(locale) {
    var group = locale.grouping === undefined || locale.thousands === undefined ? identity$1 : formatGroup(map.call(locale.grouping, Number), locale.thousands + ""),
      currencyPrefix = locale.currency === undefined ? "" : locale.currency[0] + "",
      currencySuffix = locale.currency === undefined ? "" : locale.currency[1] + "",
      decimal = locale.decimal === undefined ? "." : locale.decimal + "",
      numerals = locale.numerals === undefined ? identity$1 : formatNumerals(map.call(locale.numerals, String)),
      percent = locale.percent === undefined ? "%" : locale.percent + "",
      minus = locale.minus === undefined ? "-" : locale.minus + "",
      nan = locale.nan === undefined ? "NaN" : locale.nan + "";
    function newFormat(specifier) {
      specifier = formatSpecifier(specifier);
      var fill = specifier.fill,
        align = specifier.align,
        sign = specifier.sign,
        symbol = specifier.symbol,
        zero = specifier.zero,
        width = specifier.width,
        comma = specifier.comma,
        precision = specifier.precision,
        trim = specifier.trim,
        type = specifier.type;

      // The "n" type is an alias for ",g".
      if (type === "n") comma = true, type = "g";

      // The "" type, and any invalid type, is an alias for ".12~g".
      else if (!formatTypes[type]) precision === undefined && (precision = 12), trim = true, type = "g";

      // If zero fill is specified, padding goes after sign and before digits.
      if (zero || fill === "0" && align === "=") zero = true, fill = "0", align = "=";

      // Compute the prefix and suffix.
      // For SI-prefix, the suffix is lazily computed.
      var prefix = symbol === "$" ? currencyPrefix : symbol === "#" && /[boxX]/.test(type) ? "0" + type.toLowerCase() : "",
        suffix = symbol === "$" ? currencySuffix : /[%p]/.test(type) ? percent : "";

      // What format function should we use?
      // Is this an integer type?
      // Can this type generate exponential notation?
      var formatType = formatTypes[type],
        maybeSuffix = /[defgprs%]/.test(type);

      // Set the default precision if not specified,
      // or clamp the specified precision to the supported range.
      // For significant precision, it must be in [1, 21].
      // For fixed precision, it must be in [0, 20].
      precision = precision === undefined ? 6 : /[gprs]/.test(type) ? Math.max(1, Math.min(21, precision)) : Math.max(0, Math.min(20, precision));
      function format(value) {
        var valuePrefix = prefix,
          valueSuffix = suffix,
          i,
          n,
          c;
        if (type === "c") {
          valueSuffix = formatType(value) + valueSuffix;
          value = "";
        } else {
          value = +value;

          // Determine the sign. -0 is not less than 0, but 1 / -0 is!
          var valueNegative = value < 0 || 1 / value < 0;

          // Perform the initial formatting.
          value = isNaN(value) ? nan : formatType(Math.abs(value), precision);

          // Trim insignificant zeros.
          if (trim) value = formatTrim(value);

          // If a negative value rounds to zero after formatting, and no explicit positive sign is requested, hide the sign.
          if (valueNegative && +value === 0 && sign !== "+") valueNegative = false;

          // Compute the prefix and suffix.
          valuePrefix = (valueNegative ? sign === "(" ? sign : minus : sign === "-" || sign === "(" ? "" : sign) + valuePrefix;
          valueSuffix = (type === "s" ? prefixes[8 + prefixExponent / 3] : "") + valueSuffix + (valueNegative && sign === "(" ? ")" : "");

          // Break the formatted value into the integer “value” part that can be
          // grouped, and fractional or exponential “suffix” part that is not.
          if (maybeSuffix) {
            i = -1, n = value.length;
            while (++i < n) {
              if (c = value.charCodeAt(i), 48 > c || c > 57) {
                valueSuffix = (c === 46 ? decimal + value.slice(i + 1) : value.slice(i)) + valueSuffix;
                value = value.slice(0, i);
                break;
              }
            }
          }
        }

        // If the fill character is not "0", grouping is applied before padding.
        if (comma && !zero) value = group(value, Infinity);

        // Compute the padding.
        var length = valuePrefix.length + value.length + valueSuffix.length,
          padding = length < width ? new Array(width - length + 1).join(fill) : "";

        // If the fill character is "0", grouping is applied after padding.
        if (comma && zero) value = group(padding + value, padding.length ? width - valueSuffix.length : Infinity), padding = "";

        // Reconstruct the final output based on the desired alignment.
        switch (align) {
          case "<":
            value = valuePrefix + value + valueSuffix + padding;
            break;
          case "=":
            value = valuePrefix + padding + value + valueSuffix;
            break;
          case "^":
            value = padding.slice(0, length = padding.length >> 1) + valuePrefix + value + valueSuffix + padding.slice(length);
            break;
          default:
            value = padding + valuePrefix + value + valueSuffix;
            break;
        }
        return numerals(value);
      }
      format.toString = function () {
        return specifier + "";
      };
      return format;
    }
    function formatPrefix(specifier, value) {
      var f = newFormat((specifier = formatSpecifier(specifier), specifier.type = "f", specifier)),
        e = Math.max(-8, Math.min(8, Math.floor(exponent(value) / 3))) * 3,
        k = Math.pow(10, -e),
        prefix = prefixes[8 + e / 3];
      return function (value) {
        return f(k * value) + prefix;
      };
    }
    return {
      format: newFormat,
      formatPrefix: formatPrefix
    };
  }
  var locale;
  var format;
  var formatPrefix;
  defaultLocale({
    decimal: ".",
    thousands: ",",
    grouping: [3],
    currency: ["$", ""],
    minus: "-"
  });
  function defaultLocale(definition) {
    locale = formatLocale(definition);
    format = locale.format;
    formatPrefix = locale.formatPrefix;
    return locale;
  }
  function precisionFixed(step) {
    return Math.max(0, -exponent(Math.abs(step)));
  }
  function precisionPrefix(step, value) {
    return Math.max(0, Math.max(-8, Math.min(8, Math.floor(exponent(value) / 3))) * 3 - exponent(Math.abs(step)));
  }
  function precisionRound(step, max) {
    step = Math.abs(step), max = Math.abs(max) - step;
    return Math.max(0, exponent(max) - exponent(step)) + 1;
  }
  function tickFormat(start, stop, count, specifier) {
    var step = tickStep(start, stop, count),
      precision;
    specifier = formatSpecifier(specifier == null ? ",f" : specifier);
    switch (specifier.type) {
      case "s":
        {
          var value = Math.max(Math.abs(start), Math.abs(stop));
          if (specifier.precision == null && !isNaN(precision = precisionPrefix(step, value))) specifier.precision = precision;
          return formatPrefix(specifier, value);
        }
      case "":
      case "e":
      case "g":
      case "p":
      case "r":
        {
          if (specifier.precision == null && !isNaN(precision = precisionRound(step, Math.max(Math.abs(start), Math.abs(stop))))) specifier.precision = precision - (specifier.type === "e");
          break;
        }
      case "f":
      case "%":
        {
          if (specifier.precision == null && !isNaN(precision = precisionFixed(step))) specifier.precision = precision - (specifier.type === "%") * 2;
          break;
        }
    }
    return format(specifier);
  }
  function linearish(scale) {
    var domain = scale.domain;
    scale.ticks = function (count) {
      var d = domain();
      return ticks(d[0], d[d.length - 1], count == null ? 10 : count);
    };
    scale.tickFormat = function (count, specifier) {
      var d = domain();
      return tickFormat(d[0], d[d.length - 1], count == null ? 10 : count, specifier);
    };
    scale.nice = function (count) {
      if (count == null) count = 10;
      var d = domain(),
        i0 = 0,
        i1 = d.length - 1,
        start = d[i0],
        stop = d[i1],
        step;
      if (stop < start) {
        step = start, start = stop, stop = step;
        step = i0, i0 = i1, i1 = step;
      }
      step = tickIncrement(start, stop, count);
      if (step > 0) {
        start = Math.floor(start / step) * step;
        stop = Math.ceil(stop / step) * step;
        step = tickIncrement(start, stop, count);
      } else if (step < 0) {
        start = Math.ceil(start * step) / step;
        stop = Math.floor(stop * step) / step;
        step = tickIncrement(start, stop, count);
      }
      if (step > 0) {
        d[i0] = Math.floor(start / step) * step;
        d[i1] = Math.ceil(stop / step) * step;
        domain(d);
      } else if (step < 0) {
        d[i0] = Math.ceil(start * step) / step;
        d[i1] = Math.floor(stop * step) / step;
        domain(d);
      }
      return scale;
    };
    return scale;
  }
  function linear$1() {
    var scale = continuous();
    scale.copy = function () {
      return copy(scale, linear$1());
    };
    initRange.apply(scale, arguments);
    return linearish(scale);
  }
  var t0$1 = new Date(),
    t1$1 = new Date();
  function newInterval(floori, offseti, count, field) {
    function interval(date) {
      return floori(date = arguments.length === 0 ? new Date() : new Date(+date)), date;
    }
    interval.floor = function (date) {
      return floori(date = new Date(+date)), date;
    };
    interval.ceil = function (date) {
      return floori(date = new Date(date - 1)), offseti(date, 1), floori(date), date;
    };
    interval.round = function (date) {
      var d0 = interval(date),
        d1 = interval.ceil(date);
      return date - d0 < d1 - date ? d0 : d1;
    };
    interval.offset = function (date, step) {
      return offseti(date = new Date(+date), step == null ? 1 : Math.floor(step)), date;
    };
    interval.range = function (start, stop, step) {
      var range = [],
        previous;
      start = interval.ceil(start);
      step = step == null ? 1 : Math.floor(step);
      if (!(start < stop) || !(step > 0)) return range; // also handles Invalid Date
      do range.push(previous = new Date(+start)), offseti(start, step), floori(start); while (previous < start && start < stop);
      return range;
    };
    interval.filter = function (test) {
      return newInterval(function (date) {
        if (date >= date) while (floori(date), !test(date)) date.setTime(date - 1);
      }, function (date, step) {
        if (date >= date) {
          if (step < 0) while (++step <= 0) {
            while (offseti(date, -1), !test(date)) {} // eslint-disable-line no-empty
          } else while (--step >= 0) {
            while (offseti(date, +1), !test(date)) {} // eslint-disable-line no-empty
          }
        }
      });
    };
    if (count) {
      interval.count = function (start, end) {
        t0$1.setTime(+start), t1$1.setTime(+end);
        floori(t0$1), floori(t1$1);
        return Math.floor(count(t0$1, t1$1));
      };
      interval.every = function (step) {
        step = Math.floor(step);
        return !isFinite(step) || !(step > 0) ? null : !(step > 1) ? interval : interval.filter(field ? function (d) {
          return field(d) % step === 0;
        } : function (d) {
          return interval.count(0, d) % step === 0;
        });
      };
    }
    return interval;
  }
  var millisecond = newInterval(function () {
    // noop
  }, function (date, step) {
    date.setTime(+date + step);
  }, function (start, end) {
    return end - start;
  });

  // An optimized implementation for this simple case.
  millisecond.every = function (k) {
    k = Math.floor(k);
    if (!isFinite(k) || !(k > 0)) return null;
    if (!(k > 1)) return millisecond;
    return newInterval(function (date) {
      date.setTime(Math.floor(date / k) * k);
    }, function (date, step) {
      date.setTime(+date + step * k);
    }, function (start, end) {
      return (end - start) / k;
    });
  };
  var durationSecond = 1e3;
  var durationMinute = 6e4;
  var durationHour = 36e5;
  var durationDay = 864e5;
  var durationWeek = 6048e5;
  var second = newInterval(function (date) {
    date.setTime(date - date.getMilliseconds());
  }, function (date, step) {
    date.setTime(+date + step * durationSecond);
  }, function (start, end) {
    return (end - start) / durationSecond;
  }, function (date) {
    return date.getUTCSeconds();
  });
  var minute = newInterval(function (date) {
    date.setTime(date - date.getMilliseconds() - date.getSeconds() * durationSecond);
  }, function (date, step) {
    date.setTime(+date + step * durationMinute);
  }, function (start, end) {
    return (end - start) / durationMinute;
  }, function (date) {
    return date.getMinutes();
  });
  var hour = newInterval(function (date) {
    date.setTime(date - date.getMilliseconds() - date.getSeconds() * durationSecond - date.getMinutes() * durationMinute);
  }, function (date, step) {
    date.setTime(+date + step * durationHour);
  }, function (start, end) {
    return (end - start) / durationHour;
  }, function (date) {
    return date.getHours();
  });
  var day = newInterval(function (date) {
    date.setHours(0, 0, 0, 0);
  }, function (date, step) {
    date.setDate(date.getDate() + step);
  }, function (start, end) {
    return (end - start - (end.getTimezoneOffset() - start.getTimezoneOffset()) * durationMinute) / durationDay;
  }, function (date) {
    return date.getDate() - 1;
  });
  function weekday(i) {
    return newInterval(function (date) {
      date.setDate(date.getDate() - (date.getDay() + 7 - i) % 7);
      date.setHours(0, 0, 0, 0);
    }, function (date, step) {
      date.setDate(date.getDate() + step * 7);
    }, function (start, end) {
      return (end - start - (end.getTimezoneOffset() - start.getTimezoneOffset()) * durationMinute) / durationWeek;
    });
  }
  var sunday = weekday(0);
  var monday = weekday(1);
  var tuesday = weekday(2);
  var wednesday = weekday(3);
  var thursday = weekday(4);
  var friday = weekday(5);
  var saturday = weekday(6);
  var month = newInterval(function (date) {
    date.setDate(1);
    date.setHours(0, 0, 0, 0);
  }, function (date, step) {
    date.setMonth(date.getMonth() + step);
  }, function (start, end) {
    return end.getMonth() - start.getMonth() + (end.getFullYear() - start.getFullYear()) * 12;
  }, function (date) {
    return date.getMonth();
  });
  var year = newInterval(function (date) {
    date.setMonth(0, 1);
    date.setHours(0, 0, 0, 0);
  }, function (date, step) {
    date.setFullYear(date.getFullYear() + step);
  }, function (start, end) {
    return end.getFullYear() - start.getFullYear();
  }, function (date) {
    return date.getFullYear();
  });

  // An optimized implementation for this simple case.
  year.every = function (k) {
    return !isFinite(k = Math.floor(k)) || !(k > 0) ? null : newInterval(function (date) {
      date.setFullYear(Math.floor(date.getFullYear() / k) * k);
      date.setMonth(0, 1);
      date.setHours(0, 0, 0, 0);
    }, function (date, step) {
      date.setFullYear(date.getFullYear() + step * k);
    });
  };
  var utcMinute = newInterval(function (date) {
    date.setUTCSeconds(0, 0);
  }, function (date, step) {
    date.setTime(+date + step * durationMinute);
  }, function (start, end) {
    return (end - start) / durationMinute;
  }, function (date) {
    return date.getUTCMinutes();
  });
  var utcHour = newInterval(function (date) {
    date.setUTCMinutes(0, 0, 0);
  }, function (date, step) {
    date.setTime(+date + step * durationHour);
  }, function (start, end) {
    return (end - start) / durationHour;
  }, function (date) {
    return date.getUTCHours();
  });
  var utcDay = newInterval(function (date) {
    date.setUTCHours(0, 0, 0, 0);
  }, function (date, step) {
    date.setUTCDate(date.getUTCDate() + step);
  }, function (start, end) {
    return (end - start) / durationDay;
  }, function (date) {
    return date.getUTCDate() - 1;
  });
  function utcWeekday(i) {
    return newInterval(function (date) {
      date.setUTCDate(date.getUTCDate() - (date.getUTCDay() + 7 - i) % 7);
      date.setUTCHours(0, 0, 0, 0);
    }, function (date, step) {
      date.setUTCDate(date.getUTCDate() + step * 7);
    }, function (start, end) {
      return (end - start) / durationWeek;
    });
  }
  var utcSunday = utcWeekday(0);
  var utcMonday = utcWeekday(1);
  var utcTuesday = utcWeekday(2);
  var utcWednesday = utcWeekday(3);
  var utcThursday = utcWeekday(4);
  var utcFriday = utcWeekday(5);
  var utcSaturday = utcWeekday(6);
  var utcMonth = newInterval(function (date) {
    date.setUTCDate(1);
    date.setUTCHours(0, 0, 0, 0);
  }, function (date, step) {
    date.setUTCMonth(date.getUTCMonth() + step);
  }, function (start, end) {
    return end.getUTCMonth() - start.getUTCMonth() + (end.getUTCFullYear() - start.getUTCFullYear()) * 12;
  }, function (date) {
    return date.getUTCMonth();
  });
  var utcYear = newInterval(function (date) {
    date.setUTCMonth(0, 1);
    date.setUTCHours(0, 0, 0, 0);
  }, function (date, step) {
    date.setUTCFullYear(date.getUTCFullYear() + step);
  }, function (start, end) {
    return end.getUTCFullYear() - start.getUTCFullYear();
  }, function (date) {
    return date.getUTCFullYear();
  });

  // An optimized implementation for this simple case.
  utcYear.every = function (k) {
    return !isFinite(k = Math.floor(k)) || !(k > 0) ? null : newInterval(function (date) {
      date.setUTCFullYear(Math.floor(date.getUTCFullYear() / k) * k);
      date.setUTCMonth(0, 1);
      date.setUTCHours(0, 0, 0, 0);
    }, function (date, step) {
      date.setUTCFullYear(date.getUTCFullYear() + step * k);
    });
  };
  function localDate(d) {
    if (0 <= d.y && d.y < 100) {
      var date = new Date(-1, d.m, d.d, d.H, d.M, d.S, d.L);
      date.setFullYear(d.y);
      return date;
    }
    return new Date(d.y, d.m, d.d, d.H, d.M, d.S, d.L);
  }
  function utcDate(d) {
    if (0 <= d.y && d.y < 100) {
      var date = new Date(Date.UTC(-1, d.m, d.d, d.H, d.M, d.S, d.L));
      date.setUTCFullYear(d.y);
      return date;
    }
    return new Date(Date.UTC(d.y, d.m, d.d, d.H, d.M, d.S, d.L));
  }
  function newDate(y, m, d) {
    return {
      y: y,
      m: m,
      d: d,
      H: 0,
      M: 0,
      S: 0,
      L: 0
    };
  }
  function formatLocale$1(locale) {
    var locale_dateTime = locale.dateTime,
      locale_date = locale.date,
      locale_time = locale.time,
      locale_periods = locale.periods,
      locale_weekdays = locale.days,
      locale_shortWeekdays = locale.shortDays,
      locale_months = locale.months,
      locale_shortMonths = locale.shortMonths;
    var periodRe = formatRe(locale_periods),
      periodLookup = formatLookup(locale_periods),
      weekdayRe = formatRe(locale_weekdays),
      weekdayLookup = formatLookup(locale_weekdays),
      shortWeekdayRe = formatRe(locale_shortWeekdays),
      shortWeekdayLookup = formatLookup(locale_shortWeekdays),
      monthRe = formatRe(locale_months),
      monthLookup = formatLookup(locale_months),
      shortMonthRe = formatRe(locale_shortMonths),
      shortMonthLookup = formatLookup(locale_shortMonths);
    var formats = {
      "a": formatShortWeekday,
      "A": formatWeekday,
      "b": formatShortMonth,
      "B": formatMonth,
      "c": null,
      "d": formatDayOfMonth,
      "e": formatDayOfMonth,
      "f": formatMicroseconds,
      "H": formatHour24,
      "I": formatHour12,
      "j": formatDayOfYear,
      "L": formatMilliseconds,
      "m": formatMonthNumber,
      "M": formatMinutes,
      "p": formatPeriod,
      "q": formatQuarter,
      "Q": formatUnixTimestamp,
      "s": formatUnixTimestampSeconds,
      "S": formatSeconds,
      "u": formatWeekdayNumberMonday,
      "U": formatWeekNumberSunday,
      "V": formatWeekNumberISO,
      "w": formatWeekdayNumberSunday,
      "W": formatWeekNumberMonday,
      "x": null,
      "X": null,
      "y": formatYear,
      "Y": formatFullYear,
      "Z": formatZone,
      "%": formatLiteralPercent
    };
    var utcFormats = {
      "a": formatUTCShortWeekday,
      "A": formatUTCWeekday,
      "b": formatUTCShortMonth,
      "B": formatUTCMonth,
      "c": null,
      "d": formatUTCDayOfMonth,
      "e": formatUTCDayOfMonth,
      "f": formatUTCMicroseconds,
      "H": formatUTCHour24,
      "I": formatUTCHour12,
      "j": formatUTCDayOfYear,
      "L": formatUTCMilliseconds,
      "m": formatUTCMonthNumber,
      "M": formatUTCMinutes,
      "p": formatUTCPeriod,
      "q": formatUTCQuarter,
      "Q": formatUnixTimestamp,
      "s": formatUnixTimestampSeconds,
      "S": formatUTCSeconds,
      "u": formatUTCWeekdayNumberMonday,
      "U": formatUTCWeekNumberSunday,
      "V": formatUTCWeekNumberISO,
      "w": formatUTCWeekdayNumberSunday,
      "W": formatUTCWeekNumberMonday,
      "x": null,
      "X": null,
      "y": formatUTCYear,
      "Y": formatUTCFullYear,
      "Z": formatUTCZone,
      "%": formatLiteralPercent
    };
    var parses = {
      "a": parseShortWeekday,
      "A": parseWeekday,
      "b": parseShortMonth,
      "B": parseMonth,
      "c": parseLocaleDateTime,
      "d": parseDayOfMonth,
      "e": parseDayOfMonth,
      "f": parseMicroseconds,
      "H": parseHour24,
      "I": parseHour24,
      "j": parseDayOfYear,
      "L": parseMilliseconds,
      "m": parseMonthNumber,
      "M": parseMinutes,
      "p": parsePeriod,
      "q": parseQuarter,
      "Q": parseUnixTimestamp,
      "s": parseUnixTimestampSeconds,
      "S": parseSeconds,
      "u": parseWeekdayNumberMonday,
      "U": parseWeekNumberSunday,
      "V": parseWeekNumberISO,
      "w": parseWeekdayNumberSunday,
      "W": parseWeekNumberMonday,
      "x": parseLocaleDate,
      "X": parseLocaleTime,
      "y": parseYear,
      "Y": parseFullYear,
      "Z": parseZone,
      "%": parseLiteralPercent
    };

    // These recursive directive definitions must be deferred.
    formats.x = newFormat(locale_date, formats);
    formats.X = newFormat(locale_time, formats);
    formats.c = newFormat(locale_dateTime, formats);
    utcFormats.x = newFormat(locale_date, utcFormats);
    utcFormats.X = newFormat(locale_time, utcFormats);
    utcFormats.c = newFormat(locale_dateTime, utcFormats);
    function newFormat(specifier, formats) {
      return function (date) {
        var string = [],
          i = -1,
          j = 0,
          n = specifier.length,
          c,
          pad,
          format;
        if (!(date instanceof Date)) date = new Date(+date);
        while (++i < n) {
          if (specifier.charCodeAt(i) === 37) {
            string.push(specifier.slice(j, i));
            if ((pad = pads[c = specifier.charAt(++i)]) != null) c = specifier.charAt(++i);else pad = c === "e" ? " " : "0";
            if (format = formats[c]) c = format(date, pad);
            string.push(c);
            j = i + 1;
          }
        }
        string.push(specifier.slice(j, i));
        return string.join("");
      };
    }
    function newParse(specifier, Z) {
      return function (string) {
        var d = newDate(1900, undefined, 1),
          i = parseSpecifier(d, specifier, string += "", 0),
          week,
          day$1;
        if (i != string.length) return null;

        // If a UNIX timestamp is specified, return it.
        if ("Q" in d) return new Date(d.Q);
        if ("s" in d) return new Date(d.s * 1000 + ("L" in d ? d.L : 0));

        // If this is utcParse, never use the local timezone.
        if (Z && !("Z" in d)) d.Z = 0;

        // The am-pm flag is 0 for AM, and 1 for PM.
        if ("p" in d) d.H = d.H % 12 + d.p * 12;

        // If the month was not specified, inherit from the quarter.
        if (d.m === undefined) d.m = "q" in d ? d.q : 0;

        // Convert day-of-week and week-of-year to day-of-year.
        if ("V" in d) {
          if (d.V < 1 || d.V > 53) return null;
          if (!("w" in d)) d.w = 1;
          if ("Z" in d) {
            week = utcDate(newDate(d.y, 0, 1)), day$1 = week.getUTCDay();
            week = day$1 > 4 || day$1 === 0 ? utcMonday.ceil(week) : utcMonday(week);
            week = utcDay.offset(week, (d.V - 1) * 7);
            d.y = week.getUTCFullYear();
            d.m = week.getUTCMonth();
            d.d = week.getUTCDate() + (d.w + 6) % 7;
          } else {
            week = localDate(newDate(d.y, 0, 1)), day$1 = week.getDay();
            week = day$1 > 4 || day$1 === 0 ? monday.ceil(week) : monday(week);
            week = day.offset(week, (d.V - 1) * 7);
            d.y = week.getFullYear();
            d.m = week.getMonth();
            d.d = week.getDate() + (d.w + 6) % 7;
          }
        } else if ("W" in d || "U" in d) {
          if (!("w" in d)) d.w = "u" in d ? d.u % 7 : "W" in d ? 1 : 0;
          day$1 = "Z" in d ? utcDate(newDate(d.y, 0, 1)).getUTCDay() : localDate(newDate(d.y, 0, 1)).getDay();
          d.m = 0;
          d.d = "W" in d ? (d.w + 6) % 7 + d.W * 7 - (day$1 + 5) % 7 : d.w + d.U * 7 - (day$1 + 6) % 7;
        }

        // If a time zone is specified, all fields are interpreted as UTC and then
        // offset according to the specified time zone.
        if ("Z" in d) {
          d.H += d.Z / 100 | 0;
          d.M += d.Z % 100;
          return utcDate(d);
        }

        // Otherwise, all fields are in local time.
        return localDate(d);
      };
    }
    function parseSpecifier(d, specifier, string, j) {
      var i = 0,
        n = specifier.length,
        m = string.length,
        c,
        parse;
      while (i < n) {
        if (j >= m) return -1;
        c = specifier.charCodeAt(i++);
        if (c === 37) {
          c = specifier.charAt(i++);
          parse = parses[c in pads ? specifier.charAt(i++) : c];
          if (!parse || (j = parse(d, string, j)) < 0) return -1;
        } else if (c != string.charCodeAt(j++)) {
          return -1;
        }
      }
      return j;
    }
    function parsePeriod(d, string, i) {
      var n = periodRe.exec(string.slice(i));
      return n ? (d.p = periodLookup[n[0].toLowerCase()], i + n[0].length) : -1;
    }
    function parseShortWeekday(d, string, i) {
      var n = shortWeekdayRe.exec(string.slice(i));
      return n ? (d.w = shortWeekdayLookup[n[0].toLowerCase()], i + n[0].length) : -1;
    }
    function parseWeekday(d, string, i) {
      var n = weekdayRe.exec(string.slice(i));
      return n ? (d.w = weekdayLookup[n[0].toLowerCase()], i + n[0].length) : -1;
    }
    function parseShortMonth(d, string, i) {
      var n = shortMonthRe.exec(string.slice(i));
      return n ? (d.m = shortMonthLookup[n[0].toLowerCase()], i + n[0].length) : -1;
    }
    function parseMonth(d, string, i) {
      var n = monthRe.exec(string.slice(i));
      return n ? (d.m = monthLookup[n[0].toLowerCase()], i + n[0].length) : -1;
    }
    function parseLocaleDateTime(d, string, i) {
      return parseSpecifier(d, locale_dateTime, string, i);
    }
    function parseLocaleDate(d, string, i) {
      return parseSpecifier(d, locale_date, string, i);
    }
    function parseLocaleTime(d, string, i) {
      return parseSpecifier(d, locale_time, string, i);
    }
    function formatShortWeekday(d) {
      return locale_shortWeekdays[d.getDay()];
    }
    function formatWeekday(d) {
      return locale_weekdays[d.getDay()];
    }
    function formatShortMonth(d) {
      return locale_shortMonths[d.getMonth()];
    }
    function formatMonth(d) {
      return locale_months[d.getMonth()];
    }
    function formatPeriod(d) {
      return locale_periods[+(d.getHours() >= 12)];
    }
    function formatQuarter(d) {
      return 1 + ~~(d.getMonth() / 3);
    }
    function formatUTCShortWeekday(d) {
      return locale_shortWeekdays[d.getUTCDay()];
    }
    function formatUTCWeekday(d) {
      return locale_weekdays[d.getUTCDay()];
    }
    function formatUTCShortMonth(d) {
      return locale_shortMonths[d.getUTCMonth()];
    }
    function formatUTCMonth(d) {
      return locale_months[d.getUTCMonth()];
    }
    function formatUTCPeriod(d) {
      return locale_periods[+(d.getUTCHours() >= 12)];
    }
    function formatUTCQuarter(d) {
      return 1 + ~~(d.getUTCMonth() / 3);
    }
    return {
      format: function format(specifier) {
        var f = newFormat(specifier += "", formats);
        f.toString = function () {
          return specifier;
        };
        return f;
      },
      parse: function parse(specifier) {
        var p = newParse(specifier += "", false);
        p.toString = function () {
          return specifier;
        };
        return p;
      },
      utcFormat: function utcFormat(specifier) {
        var f = newFormat(specifier += "", utcFormats);
        f.toString = function () {
          return specifier;
        };
        return f;
      },
      utcParse: function utcParse(specifier) {
        var p = newParse(specifier += "", true);
        p.toString = function () {
          return specifier;
        };
        return p;
      }
    };
  }
  var pads = {
      "-": "",
      "_": " ",
      "0": "0"
    },
    numberRe = /^\s*\d+/,
    // note: ignores next directive
    percentRe = /^%/,
    requoteRe = /[\\^$*+?|[\]().{}]/g;
  function pad(value, fill, width) {
    var sign = value < 0 ? "-" : "",
      string = (sign ? -value : value) + "",
      length = string.length;
    return sign + (length < width ? new Array(width - length + 1).join(fill) + string : string);
  }
  function requote(s) {
    return s.replace(requoteRe, "\\$&");
  }
  function formatRe(names) {
    return new RegExp("^(?:" + names.map(requote).join("|") + ")", "i");
  }
  function formatLookup(names) {
    var map = {},
      i = -1,
      n = names.length;
    while (++i < n) map[names[i].toLowerCase()] = i;
    return map;
  }
  function parseWeekdayNumberSunday(d, string, i) {
    var n = numberRe.exec(string.slice(i, i + 1));
    return n ? (d.w = +n[0], i + n[0].length) : -1;
  }
  function parseWeekdayNumberMonday(d, string, i) {
    var n = numberRe.exec(string.slice(i, i + 1));
    return n ? (d.u = +n[0], i + n[0].length) : -1;
  }
  function parseWeekNumberSunday(d, string, i) {
    var n = numberRe.exec(string.slice(i, i + 2));
    return n ? (d.U = +n[0], i + n[0].length) : -1;
  }
  function parseWeekNumberISO(d, string, i) {
    var n = numberRe.exec(string.slice(i, i + 2));
    return n ? (d.V = +n[0], i + n[0].length) : -1;
  }
  function parseWeekNumberMonday(d, string, i) {
    var n = numberRe.exec(string.slice(i, i + 2));
    return n ? (d.W = +n[0], i + n[0].length) : -1;
  }
  function parseFullYear(d, string, i) {
    var n = numberRe.exec(string.slice(i, i + 4));
    return n ? (d.y = +n[0], i + n[0].length) : -1;
  }
  function parseYear(d, string, i) {
    var n = numberRe.exec(string.slice(i, i + 2));
    return n ? (d.y = +n[0] + (+n[0] > 68 ? 1900 : 2000), i + n[0].length) : -1;
  }
  function parseZone(d, string, i) {
    var n = /^(Z)|([+-]\d\d)(?::?(\d\d))?/.exec(string.slice(i, i + 6));
    return n ? (d.Z = n[1] ? 0 : -(n[2] + (n[3] || "00")), i + n[0].length) : -1;
  }
  function parseQuarter(d, string, i) {
    var n = numberRe.exec(string.slice(i, i + 1));
    return n ? (d.q = n[0] * 3 - 3, i + n[0].length) : -1;
  }
  function parseMonthNumber(d, string, i) {
    var n = numberRe.exec(string.slice(i, i + 2));
    return n ? (d.m = n[0] - 1, i + n[0].length) : -1;
  }
  function parseDayOfMonth(d, string, i) {
    var n = numberRe.exec(string.slice(i, i + 2));
    return n ? (d.d = +n[0], i + n[0].length) : -1;
  }
  function parseDayOfYear(d, string, i) {
    var n = numberRe.exec(string.slice(i, i + 3));
    return n ? (d.m = 0, d.d = +n[0], i + n[0].length) : -1;
  }
  function parseHour24(d, string, i) {
    var n = numberRe.exec(string.slice(i, i + 2));
    return n ? (d.H = +n[0], i + n[0].length) : -1;
  }
  function parseMinutes(d, string, i) {
    var n = numberRe.exec(string.slice(i, i + 2));
    return n ? (d.M = +n[0], i + n[0].length) : -1;
  }
  function parseSeconds(d, string, i) {
    var n = numberRe.exec(string.slice(i, i + 2));
    return n ? (d.S = +n[0], i + n[0].length) : -1;
  }
  function parseMilliseconds(d, string, i) {
    var n = numberRe.exec(string.slice(i, i + 3));
    return n ? (d.L = +n[0], i + n[0].length) : -1;
  }
  function parseMicroseconds(d, string, i) {
    var n = numberRe.exec(string.slice(i, i + 6));
    return n ? (d.L = Math.floor(n[0] / 1000), i + n[0].length) : -1;
  }
  function parseLiteralPercent(d, string, i) {
    var n = percentRe.exec(string.slice(i, i + 1));
    return n ? i + n[0].length : -1;
  }
  function parseUnixTimestamp(d, string, i) {
    var n = numberRe.exec(string.slice(i));
    return n ? (d.Q = +n[0], i + n[0].length) : -1;
  }
  function parseUnixTimestampSeconds(d, string, i) {
    var n = numberRe.exec(string.slice(i));
    return n ? (d.s = +n[0], i + n[0].length) : -1;
  }
  function formatDayOfMonth(d, p) {
    return pad(d.getDate(), p, 2);
  }
  function formatHour24(d, p) {
    return pad(d.getHours(), p, 2);
  }
  function formatHour12(d, p) {
    return pad(d.getHours() % 12 || 12, p, 2);
  }
  function formatDayOfYear(d, p) {
    return pad(1 + day.count(year(d), d), p, 3);
  }
  function formatMilliseconds(d, p) {
    return pad(d.getMilliseconds(), p, 3);
  }
  function formatMicroseconds(d, p) {
    return formatMilliseconds(d, p) + "000";
  }
  function formatMonthNumber(d, p) {
    return pad(d.getMonth() + 1, p, 2);
  }
  function formatMinutes(d, p) {
    return pad(d.getMinutes(), p, 2);
  }
  function formatSeconds(d, p) {
    return pad(d.getSeconds(), p, 2);
  }
  function formatWeekdayNumberMonday(d) {
    var day = d.getDay();
    return day === 0 ? 7 : day;
  }
  function formatWeekNumberSunday(d, p) {
    return pad(sunday.count(year(d) - 1, d), p, 2);
  }
  function formatWeekNumberISO(d, p) {
    var day = d.getDay();
    d = day >= 4 || day === 0 ? thursday(d) : thursday.ceil(d);
    return pad(thursday.count(year(d), d) + (year(d).getDay() === 4), p, 2);
  }
  function formatWeekdayNumberSunday(d) {
    return d.getDay();
  }
  function formatWeekNumberMonday(d, p) {
    return pad(monday.count(year(d) - 1, d), p, 2);
  }
  function formatYear(d, p) {
    return pad(d.getFullYear() % 100, p, 2);
  }
  function formatFullYear(d, p) {
    return pad(d.getFullYear() % 10000, p, 4);
  }
  function formatZone(d) {
    var z = d.getTimezoneOffset();
    return (z > 0 ? "-" : (z *= -1, "+")) + pad(z / 60 | 0, "0", 2) + pad(z % 60, "0", 2);
  }
  function formatUTCDayOfMonth(d, p) {
    return pad(d.getUTCDate(), p, 2);
  }
  function formatUTCHour24(d, p) {
    return pad(d.getUTCHours(), p, 2);
  }
  function formatUTCHour12(d, p) {
    return pad(d.getUTCHours() % 12 || 12, p, 2);
  }
  function formatUTCDayOfYear(d, p) {
    return pad(1 + utcDay.count(utcYear(d), d), p, 3);
  }
  function formatUTCMilliseconds(d, p) {
    return pad(d.getUTCMilliseconds(), p, 3);
  }
  function formatUTCMicroseconds(d, p) {
    return formatUTCMilliseconds(d, p) + "000";
  }
  function formatUTCMonthNumber(d, p) {
    return pad(d.getUTCMonth() + 1, p, 2);
  }
  function formatUTCMinutes(d, p) {
    return pad(d.getUTCMinutes(), p, 2);
  }
  function formatUTCSeconds(d, p) {
    return pad(d.getUTCSeconds(), p, 2);
  }
  function formatUTCWeekdayNumberMonday(d) {
    var dow = d.getUTCDay();
    return dow === 0 ? 7 : dow;
  }
  function formatUTCWeekNumberSunday(d, p) {
    return pad(utcSunday.count(utcYear(d) - 1, d), p, 2);
  }
  function formatUTCWeekNumberISO(d, p) {
    var day = d.getUTCDay();
    d = day >= 4 || day === 0 ? utcThursday(d) : utcThursday.ceil(d);
    return pad(utcThursday.count(utcYear(d), d) + (utcYear(d).getUTCDay() === 4), p, 2);
  }
  function formatUTCWeekdayNumberSunday(d) {
    return d.getUTCDay();
  }
  function formatUTCWeekNumberMonday(d, p) {
    return pad(utcMonday.count(utcYear(d) - 1, d), p, 2);
  }
  function formatUTCYear(d, p) {
    return pad(d.getUTCFullYear() % 100, p, 2);
  }
  function formatUTCFullYear(d, p) {
    return pad(d.getUTCFullYear() % 10000, p, 4);
  }
  function formatUTCZone() {
    return "+0000";
  }
  function formatLiteralPercent() {
    return "%";
  }
  function formatUnixTimestamp(d) {
    return +d;
  }
  function formatUnixTimestampSeconds(d) {
    return Math.floor(+d / 1000);
  }
  var locale$1;
  var timeFormat;
  var timeParse;
  var utcFormat;
  var utcParse;
  defaultLocale$1({
    dateTime: "%x, %X",
    date: "%-m/%-d/%Y",
    time: "%-I:%M:%S %p",
    periods: ["AM", "PM"],
    days: ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
    shortDays: ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
    months: ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
    shortMonths: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
  });
  function defaultLocale$1(definition) {
    locale$1 = formatLocale$1(definition);
    timeFormat = locale$1.format;
    timeParse = locale$1.parse;
    utcFormat = locale$1.utcFormat;
    utcParse = locale$1.utcParse;
    return locale$1;
  }
  var isoSpecifier = "%Y-%m-%dT%H:%M:%S.%LZ";
  function formatIsoNative(date) {
    return date.toISOString();
  }
  var formatIso = Date.prototype.toISOString ? formatIsoNative : utcFormat(isoSpecifier);
  function parseIsoNative(string) {
    var date = new Date(string);
    return isNaN(date) ? null : date;
  }
  var parseIso = +new Date("2000-01-01T00:00:00.000Z") ? parseIsoNative : utcParse(isoSpecifier);
  var noop = {
    value: function value() {}
  };
  function dispatch() {
    for (var i = 0, n = arguments.length, _ = {}, t; i < n; ++i) {
      if (!(t = arguments[i] + "") || t in _ || /[\s.]/.test(t)) throw new Error("illegal type: " + t);
      _[t] = [];
    }
    return new Dispatch(_);
  }
  function Dispatch(_) {
    this._ = _;
  }
  function parseTypenames(typenames, types) {
    return typenames.trim().split(/^|\s+/).map(function (t) {
      var name = "",
        i = t.indexOf(".");
      if (i >= 0) name = t.slice(i + 1), t = t.slice(0, i);
      if (t && !types.hasOwnProperty(t)) throw new Error("unknown type: " + t);
      return {
        type: t,
        name: name
      };
    });
  }
  Dispatch.prototype = dispatch.prototype = {
    constructor: Dispatch,
    on: function on(typename, callback) {
      var _ = this._,
        T = parseTypenames(typename + "", _),
        t,
        i = -1,
        n = T.length;

      // If no callback was specified, return the callback of the given type and name.
      if (arguments.length < 2) {
        while (++i < n) if ((t = (typename = T[i]).type) && (t = get(_[t], typename.name))) return t;
        return;
      }

      // If a type was specified, set the callback for the given type and name.
      // Otherwise, if a null callback was specified, remove callbacks of the given name.
      if (callback != null && typeof callback !== "function") throw new Error("invalid callback: " + callback);
      while (++i < n) {
        if (t = (typename = T[i]).type) _[t] = set(_[t], typename.name, callback);else if (callback == null) for (t in _) _[t] = set(_[t], typename.name, null);
      }
      return this;
    },
    copy: function copy() {
      var copy = {},
        _ = this._;
      for (var t in _) copy[t] = _[t].slice();
      return new Dispatch(copy);
    },
    call: function call(type, that) {
      if ((n = arguments.length - 2) > 0) for (var args = new Array(n), i = 0, n, t; i < n; ++i) args[i] = arguments[i + 2];
      if (!this._.hasOwnProperty(type)) throw new Error("unknown type: " + type);
      for (t = this._[type], i = 0, n = t.length; i < n; ++i) t[i].value.apply(that, args);
    },
    apply: function apply(type, that, args) {
      if (!this._.hasOwnProperty(type)) throw new Error("unknown type: " + type);
      for (var t = this._[type], i = 0, n = t.length; i < n; ++i) t[i].value.apply(that, args);
    }
  };
  function get(type, name) {
    for (var i = 0, n = type.length, c; i < n; ++i) {
      if ((c = type[i]).name === name) {
        return c.value;
      }
    }
  }
  function set(type, name, callback) {
    for (var i = 0, n = type.length; i < n; ++i) {
      if (type[i].name === name) {
        type[i] = noop, type = type.slice(0, i).concat(type.slice(i + 1));
        break;
      }
    }
    if (callback != null) type.push({
      name: name,
      value: callback
    });
    return type;
  }
  var xhtml = "http://www.w3.org/1999/xhtml";
  var namespaces = {
    svg: "http://www.w3.org/2000/svg",
    xhtml: xhtml,
    xlink: "http://www.w3.org/1999/xlink",
    xml: "http://www.w3.org/XML/1998/namespace",
    xmlns: "http://www.w3.org/2000/xmlns/"
  };
  function namespace(name) {
    var prefix = name += "",
      i = prefix.indexOf(":");
    if (i >= 0 && (prefix = name.slice(0, i)) !== "xmlns") name = name.slice(i + 1);
    return namespaces.hasOwnProperty(prefix) ? {
      space: namespaces[prefix],
      local: name
    } : name;
  }
  function creatorInherit(name) {
    return function () {
      var document = this.ownerDocument,
        uri = this.namespaceURI;
      return uri === xhtml && document.documentElement.namespaceURI === xhtml ? document.createElement(name) : document.createElementNS(uri, name);
    };
  }
  function creatorFixed(fullname) {
    return function () {
      return this.ownerDocument.createElementNS(fullname.space, fullname.local);
    };
  }
  function creator(name) {
    var fullname = namespace(name);
    return (fullname.local ? creatorFixed : creatorInherit)(fullname);
  }
  function none() {}
  function selector(selector) {
    return selector == null ? none : function () {
      return this.querySelector(selector);
    };
  }
  function selection_select(select) {
    if (typeof select !== "function") select = selector(select);
    for (var groups = this._groups, m = groups.length, subgroups = new Array(m), j = 0; j < m; ++j) {
      for (var group = groups[j], n = group.length, subgroup = subgroups[j] = new Array(n), node, subnode, i = 0; i < n; ++i) {
        if ((node = group[i]) && (subnode = select.call(node, node.__data__, i, group))) {
          if ("__data__" in node) subnode.__data__ = node.__data__;
          subgroup[i] = subnode;
        }
      }
    }
    return new Selection(subgroups, this._parents);
  }
  function empty() {
    return [];
  }
  function selectorAll(selector) {
    return selector == null ? empty : function () {
      return this.querySelectorAll(selector);
    };
  }
  function selection_selectAll(select) {
    if (typeof select !== "function") select = selectorAll(select);
    for (var groups = this._groups, m = groups.length, subgroups = [], parents = [], j = 0; j < m; ++j) {
      for (var group = groups[j], n = group.length, node, i = 0; i < n; ++i) {
        if (node = group[i]) {
          subgroups.push(select.call(node, node.__data__, i, group));
          parents.push(node);
        }
      }
    }
    return new Selection(subgroups, parents);
  }
  function matcher(selector) {
    return function () {
      return this.matches(selector);
    };
  }
  function selection_filter(match) {
    if (typeof match !== "function") match = matcher(match);
    for (var groups = this._groups, m = groups.length, subgroups = new Array(m), j = 0; j < m; ++j) {
      for (var group = groups[j], n = group.length, subgroup = subgroups[j] = [], node, i = 0; i < n; ++i) {
        if ((node = group[i]) && match.call(node, node.__data__, i, group)) {
          subgroup.push(node);
        }
      }
    }
    return new Selection(subgroups, this._parents);
  }
  function sparse(update) {
    return new Array(update.length);
  }
  function selection_enter() {
    return new Selection(this._enter || this._groups.map(sparse), this._parents);
  }
  function EnterNode(parent, datum) {
    this.ownerDocument = parent.ownerDocument;
    this.namespaceURI = parent.namespaceURI;
    this._next = null;
    this._parent = parent;
    this.__data__ = datum;
  }
  EnterNode.prototype = {
    constructor: EnterNode,
    appendChild: function appendChild(child) {
      return this._parent.insertBefore(child, this._next);
    },
    insertBefore: function insertBefore(child, next) {
      return this._parent.insertBefore(child, next);
    },
    querySelector: function querySelector(selector) {
      return this._parent.querySelector(selector);
    },
    querySelectorAll: function querySelectorAll(selector) {
      return this._parent.querySelectorAll(selector);
    }
  };
  function constant$2(x) {
    return function () {
      return x;
    };
  }
  var keyPrefix = "$"; // Protect against keys like “__proto__”.

  function bindIndex(parent, group, enter, update, exit, data) {
    var i = 0,
      node,
      groupLength = group.length,
      dataLength = data.length;

    // Put any non-null nodes that fit into update.
    // Put any null nodes into enter.
    // Put any remaining data into enter.
    for (; i < dataLength; ++i) {
      if (node = group[i]) {
        node.__data__ = data[i];
        update[i] = node;
      } else {
        enter[i] = new EnterNode(parent, data[i]);
      }
    }

    // Put any non-null nodes that don’t fit into exit.
    for (; i < groupLength; ++i) {
      if (node = group[i]) {
        exit[i] = node;
      }
    }
  }
  function bindKey(parent, group, enter, update, exit, data, key) {
    var i,
      node,
      nodeByKeyValue = {},
      groupLength = group.length,
      dataLength = data.length,
      keyValues = new Array(groupLength),
      keyValue;

    // Compute the key for each node.
    // If multiple nodes have the same key, the duplicates are added to exit.
    for (i = 0; i < groupLength; ++i) {
      if (node = group[i]) {
        keyValues[i] = keyValue = keyPrefix + key.call(node, node.__data__, i, group);
        if (keyValue in nodeByKeyValue) {
          exit[i] = node;
        } else {
          nodeByKeyValue[keyValue] = node;
        }
      }
    }

    // Compute the key for each datum.
    // If there a node associated with this key, join and add it to update.
    // If there is not (or the key is a duplicate), add it to enter.
    for (i = 0; i < dataLength; ++i) {
      keyValue = keyPrefix + key.call(parent, data[i], i, data);
      if (node = nodeByKeyValue[keyValue]) {
        update[i] = node;
        node.__data__ = data[i];
        nodeByKeyValue[keyValue] = null;
      } else {
        enter[i] = new EnterNode(parent, data[i]);
      }
    }

    // Add any remaining nodes that were not bound to data to exit.
    for (i = 0; i < groupLength; ++i) {
      if ((node = group[i]) && nodeByKeyValue[keyValues[i]] === node) {
        exit[i] = node;
      }
    }
  }
  function selection_data(value, key) {
    if (!value) {
      data = new Array(this.size()), j = -1;
      this.each(function (d) {
        data[++j] = d;
      });
      return data;
    }
    var bind = key ? bindKey : bindIndex,
      parents = this._parents,
      groups = this._groups;
    if (typeof value !== "function") value = constant$2(value);
    for (var m = groups.length, update = new Array(m), enter = new Array(m), exit = new Array(m), j = 0; j < m; ++j) {
      var parent = parents[j],
        group = groups[j],
        groupLength = group.length,
        data = value.call(parent, parent && parent.__data__, j, parents),
        dataLength = data.length,
        enterGroup = enter[j] = new Array(dataLength),
        updateGroup = update[j] = new Array(dataLength),
        exitGroup = exit[j] = new Array(groupLength);
      bind(parent, group, enterGroup, updateGroup, exitGroup, data, key);

      // Now connect the enter nodes to their following update node, such that
      // appendChild can insert the materialized enter node before this node,
      // rather than at the end of the parent node.
      for (var i0 = 0, i1 = 0, previous, next; i0 < dataLength; ++i0) {
        if (previous = enterGroup[i0]) {
          if (i0 >= i1) i1 = i0 + 1;
          while (!(next = updateGroup[i1]) && ++i1 < dataLength);
          previous._next = next || null;
        }
      }
    }
    update = new Selection(update, parents);
    update._enter = enter;
    update._exit = exit;
    return update;
  }
  function selection_exit() {
    return new Selection(this._exit || this._groups.map(sparse), this._parents);
  }
  function selection_join(onenter, onupdate, onexit) {
    var enter = this.enter(),
      update = this,
      exit = this.exit();
    enter = typeof onenter === "function" ? onenter(enter) : enter.append(onenter + "");
    if (onupdate != null) update = onupdate(update);
    if (onexit == null) exit.remove();else onexit(exit);
    return enter && update ? enter.merge(update).order() : update;
  }
  function selection_merge(selection) {
    for (var groups0 = this._groups, groups1 = selection._groups, m0 = groups0.length, m1 = groups1.length, m = Math.min(m0, m1), merges = new Array(m0), j = 0; j < m; ++j) {
      for (var group0 = groups0[j], group1 = groups1[j], n = group0.length, merge = merges[j] = new Array(n), node, i = 0; i < n; ++i) {
        if (node = group0[i] || group1[i]) {
          merge[i] = node;
        }
      }
    }
    for (; j < m0; ++j) {
      merges[j] = groups0[j];
    }
    return new Selection(merges, this._parents);
  }
  function selection_order() {
    for (var groups = this._groups, j = -1, m = groups.length; ++j < m;) {
      for (var group = groups[j], i = group.length - 1, next = group[i], node; --i >= 0;) {
        if (node = group[i]) {
          if (next && node.compareDocumentPosition(next) ^ 4) next.parentNode.insertBefore(node, next);
          next = node;
        }
      }
    }
    return this;
  }
  function selection_sort(compare) {
    if (!compare) compare = ascending$1;
    function compareNode(a, b) {
      return a && b ? compare(a.__data__, b.__data__) : !a - !b;
    }
    for (var groups = this._groups, m = groups.length, sortgroups = new Array(m), j = 0; j < m; ++j) {
      for (var group = groups[j], n = group.length, sortgroup = sortgroups[j] = new Array(n), node, i = 0; i < n; ++i) {
        if (node = group[i]) {
          sortgroup[i] = node;
        }
      }
      sortgroup.sort(compareNode);
    }
    return new Selection(sortgroups, this._parents).order();
  }
  function ascending$1(a, b) {
    return a < b ? -1 : a > b ? 1 : a >= b ? 0 : NaN;
  }
  function selection_call() {
    var callback = arguments[0];
    arguments[0] = this;
    callback.apply(null, arguments);
    return this;
  }
  function selection_nodes() {
    var nodes = new Array(this.size()),
      i = -1;
    this.each(function () {
      nodes[++i] = this;
    });
    return nodes;
  }
  function selection_node() {
    for (var groups = this._groups, j = 0, m = groups.length; j < m; ++j) {
      for (var group = groups[j], i = 0, n = group.length; i < n; ++i) {
        var node = group[i];
        if (node) return node;
      }
    }
    return null;
  }
  function selection_size() {
    var size = 0;
    this.each(function () {
      ++size;
    });
    return size;
  }
  function selection_empty() {
    return !this.node();
  }
  function selection_each(callback) {
    for (var groups = this._groups, j = 0, m = groups.length; j < m; ++j) {
      for (var group = groups[j], i = 0, n = group.length, node; i < n; ++i) {
        if (node = group[i]) callback.call(node, node.__data__, i, group);
      }
    }
    return this;
  }
  function attrRemove(name) {
    return function () {
      this.removeAttribute(name);
    };
  }
  function attrRemoveNS(fullname) {
    return function () {
      this.removeAttributeNS(fullname.space, fullname.local);
    };
  }
  function attrConstant(name, value) {
    return function () {
      this.setAttribute(name, value);
    };
  }
  function attrConstantNS(fullname, value) {
    return function () {
      this.setAttributeNS(fullname.space, fullname.local, value);
    };
  }
  function attrFunction(name, value) {
    return function () {
      var v = value.apply(this, arguments);
      if (v == null) this.removeAttribute(name);else this.setAttribute(name, v);
    };
  }
  function attrFunctionNS(fullname, value) {
    return function () {
      var v = value.apply(this, arguments);
      if (v == null) this.removeAttributeNS(fullname.space, fullname.local);else this.setAttributeNS(fullname.space, fullname.local, v);
    };
  }
  function selection_attr(name, value) {
    var fullname = namespace(name);
    if (arguments.length < 2) {
      var node = this.node();
      return fullname.local ? node.getAttributeNS(fullname.space, fullname.local) : node.getAttribute(fullname);
    }
    return this.each((value == null ? fullname.local ? attrRemoveNS : attrRemove : typeof value === "function" ? fullname.local ? attrFunctionNS : attrFunction : fullname.local ? attrConstantNS : attrConstant)(fullname, value));
  }
  function defaultView(node) {
    return node.ownerDocument && node.ownerDocument.defaultView // node is a Node
    || node.document && node // node is a Window
    || node.defaultView; // node is a Document
  }
  function styleRemove(name) {
    return function () {
      this.style.removeProperty(name);
    };
  }
  function styleConstant(name, value, priority) {
    return function () {
      this.style.setProperty(name, value, priority);
    };
  }
  function styleFunction(name, value, priority) {
    return function () {
      var v = value.apply(this, arguments);
      if (v == null) this.style.removeProperty(name);else this.style.setProperty(name, v, priority);
    };
  }
  function selection_style(name, value, priority) {
    return arguments.length > 1 ? this.each((value == null ? styleRemove : typeof value === "function" ? styleFunction : styleConstant)(name, value, priority == null ? "" : priority)) : styleValue(this.node(), name);
  }
  function styleValue(node, name) {
    return node.style.getPropertyValue(name) || defaultView(node).getComputedStyle(node, null).getPropertyValue(name);
  }
  function propertyRemove(name) {
    return function () {
      delete this[name];
    };
  }
  function propertyConstant(name, value) {
    return function () {
      this[name] = value;
    };
  }
  function propertyFunction(name, value) {
    return function () {
      var v = value.apply(this, arguments);
      if (v == null) delete this[name];else this[name] = v;
    };
  }
  function selection_property(name, value) {
    return arguments.length > 1 ? this.each((value == null ? propertyRemove : typeof value === "function" ? propertyFunction : propertyConstant)(name, value)) : this.node()[name];
  }
  function classArray(string) {
    return string.trim().split(/^|\s+/);
  }
  function classList(node) {
    return node.classList || new ClassList(node);
  }
  function ClassList(node) {
    this._node = node;
    this._names = classArray(node.getAttribute("class") || "");
  }
  ClassList.prototype = {
    add: function add(name) {
      var i = this._names.indexOf(name);
      if (i < 0) {
        this._names.push(name);
        this._node.setAttribute("class", this._names.join(" "));
      }
    },
    remove: function remove(name) {
      var i = this._names.indexOf(name);
      if (i >= 0) {
        this._names.splice(i, 1);
        this._node.setAttribute("class", this._names.join(" "));
      }
    },
    contains: function contains(name) {
      return this._names.indexOf(name) >= 0;
    }
  };
  function classedAdd(node, names) {
    var list = classList(node),
      i = -1,
      n = names.length;
    while (++i < n) list.add(names[i]);
  }
  function classedRemove(node, names) {
    var list = classList(node),
      i = -1,
      n = names.length;
    while (++i < n) list.remove(names[i]);
  }
  function classedTrue(names) {
    return function () {
      classedAdd(this, names);
    };
  }
  function classedFalse(names) {
    return function () {
      classedRemove(this, names);
    };
  }
  function classedFunction(names, value) {
    return function () {
      (value.apply(this, arguments) ? classedAdd : classedRemove)(this, names);
    };
  }
  function selection_classed(name, value) {
    var names = classArray(name + "");
    if (arguments.length < 2) {
      var list = classList(this.node()),
        i = -1,
        n = names.length;
      while (++i < n) if (!list.contains(names[i])) return false;
      return true;
    }
    return this.each((typeof value === "function" ? classedFunction : value ? classedTrue : classedFalse)(names, value));
  }
  function textRemove() {
    this.textContent = "";
  }
  function textConstant(value) {
    return function () {
      this.textContent = value;
    };
  }
  function textFunction(value) {
    return function () {
      var v = value.apply(this, arguments);
      this.textContent = v == null ? "" : v;
    };
  }
  function selection_text(value) {
    return arguments.length ? this.each(value == null ? textRemove : (typeof value === "function" ? textFunction : textConstant)(value)) : this.node().textContent;
  }
  function htmlRemove() {
    this.innerHTML = "";
  }
  function htmlConstant(value) {
    return function () {
      this.innerHTML = value;
    };
  }
  function htmlFunction(value) {
    return function () {
      var v = value.apply(this, arguments);
      this.innerHTML = v == null ? "" : v;
    };
  }
  function selection_html(value) {
    return arguments.length ? this.each(value == null ? htmlRemove : (typeof value === "function" ? htmlFunction : htmlConstant)(value)) : this.node().innerHTML;
  }
  function raise() {
    if (this.nextSibling) this.parentNode.appendChild(this);
  }
  function selection_raise() {
    return this.each(raise);
  }
  function lower() {
    if (this.previousSibling) this.parentNode.insertBefore(this, this.parentNode.firstChild);
  }
  function selection_lower() {
    return this.each(lower);
  }
  function selection_append(name) {
    var create = typeof name === "function" ? name : creator(name);
    return this.select(function () {
      return this.appendChild(create.apply(this, arguments));
    });
  }
  function constantNull() {
    return null;
  }
  function selection_insert(name, before) {
    var create = typeof name === "function" ? name : creator(name),
      select = before == null ? constantNull : typeof before === "function" ? before : selector(before);
    return this.select(function () {
      return this.insertBefore(create.apply(this, arguments), select.apply(this, arguments) || null);
    });
  }
  function remove() {
    var parent = this.parentNode;
    if (parent) parent.removeChild(this);
  }
  function selection_remove() {
    return this.each(remove);
  }
  function selection_cloneShallow() {
    var clone = this.cloneNode(false),
      parent = this.parentNode;
    return parent ? parent.insertBefore(clone, this.nextSibling) : clone;
  }
  function selection_cloneDeep() {
    var clone = this.cloneNode(true),
      parent = this.parentNode;
    return parent ? parent.insertBefore(clone, this.nextSibling) : clone;
  }
  function selection_clone(deep) {
    return this.select(deep ? selection_cloneDeep : selection_cloneShallow);
  }
  function selection_datum(value) {
    return arguments.length ? this.property("__data__", value) : this.node().__data__;
  }
  var filterEvents = {};
  var event = null;
  if (typeof document !== "undefined") {
    var element = document.documentElement;
    if (!("onmouseenter" in element)) {
      filterEvents = {
        mouseenter: "mouseover",
        mouseleave: "mouseout"
      };
    }
  }
  function filterContextListener(listener, index, group) {
    listener = contextListener(listener, index, group);
    return function (event) {
      var related = event.relatedTarget;
      if (!related || related !== this && !(related.compareDocumentPosition(this) & 8)) {
        listener.call(this, event);
      }
    };
  }
  function contextListener(listener, index, group) {
    return function (event1) {
      var event0 = event; // Events can be reentrant (e.g., focus).
      event = event1;
      try {
        listener.call(this, this.__data__, index, group);
      } finally {
        event = event0;
      }
    };
  }
  function parseTypenames$1(typenames) {
    return typenames.trim().split(/^|\s+/).map(function (t) {
      var name = "",
        i = t.indexOf(".");
      if (i >= 0) name = t.slice(i + 1), t = t.slice(0, i);
      return {
        type: t,
        name: name
      };
    });
  }
  function onRemove(typename) {
    return function () {
      var on = this.__on;
      if (!on) return;
      for (var j = 0, i = -1, m = on.length, o; j < m; ++j) {
        if (o = on[j], (!typename.type || o.type === typename.type) && o.name === typename.name) {
          this.removeEventListener(o.type, o.listener, o.capture);
        } else {
          on[++i] = o;
        }
      }
      if (++i) on.length = i;else delete this.__on;
    };
  }
  function onAdd(typename, value, capture) {
    var wrap = filterEvents.hasOwnProperty(typename.type) ? filterContextListener : contextListener;
    return function (d, i, group) {
      var on = this.__on,
        o,
        listener = wrap(value, i, group);
      if (on) for (var j = 0, m = on.length; j < m; ++j) {
        if ((o = on[j]).type === typename.type && o.name === typename.name) {
          this.removeEventListener(o.type, o.listener, o.capture);
          this.addEventListener(o.type, o.listener = listener, o.capture = capture);
          o.value = value;
          return;
        }
      }
      this.addEventListener(typename.type, listener, capture);
      o = {
        type: typename.type,
        name: typename.name,
        value: value,
        listener: listener,
        capture: capture
      };
      if (!on) this.__on = [o];else on.push(o);
    };
  }
  function selection_on(typename, value, capture) {
    var typenames = parseTypenames$1(typename + ""),
      i,
      n = typenames.length,
      t;
    if (arguments.length < 2) {
      var on = this.node().__on;
      if (on) for (var j = 0, m = on.length, o; j < m; ++j) {
        for (i = 0, o = on[j]; i < n; ++i) {
          if ((t = typenames[i]).type === o.type && t.name === o.name) {
            return o.value;
          }
        }
      }
      return;
    }
    on = value ? onAdd : onRemove;
    if (capture == null) capture = false;
    for (i = 0; i < n; ++i) this.each(on(typenames[i], value, capture));
    return this;
  }
  function customEvent(event1, listener, that, args) {
    var event0 = event;
    event1.sourceEvent = event;
    event = event1;
    try {
      return listener.apply(that, args);
    } finally {
      event = event0;
    }
  }
  function dispatchEvent(node, type, params) {
    var window = defaultView(node),
      event = window.CustomEvent;
    if (typeof event === "function") {
      event = new event(type, params);
    } else {
      event = window.document.createEvent("Event");
      if (params) event.initEvent(type, params.bubbles, params.cancelable), event.detail = params.detail;else event.initEvent(type, false, false);
    }
    node.dispatchEvent(event);
  }
  function dispatchConstant(type, params) {
    return function () {
      return dispatchEvent(this, type, params);
    };
  }
  function dispatchFunction(type, params) {
    return function () {
      return dispatchEvent(this, type, params.apply(this, arguments));
    };
  }
  function selection_dispatch(type, params) {
    return this.each((typeof params === "function" ? dispatchFunction : dispatchConstant)(type, params));
  }
  var root = [null];
  function Selection(groups, parents) {
    this._groups = groups;
    this._parents = parents;
  }
  function selection() {
    return new Selection([[document.documentElement]], root);
  }
  Selection.prototype = selection.prototype = {
    constructor: Selection,
    select: selection_select,
    selectAll: selection_selectAll,
    filter: selection_filter,
    data: selection_data,
    enter: selection_enter,
    exit: selection_exit,
    join: selection_join,
    merge: selection_merge,
    order: selection_order,
    sort: selection_sort,
    call: selection_call,
    nodes: selection_nodes,
    node: selection_node,
    size: selection_size,
    empty: selection_empty,
    each: selection_each,
    attr: selection_attr,
    style: selection_style,
    property: selection_property,
    classed: selection_classed,
    text: selection_text,
    html: selection_html,
    raise: selection_raise,
    lower: selection_lower,
    append: selection_append,
    insert: selection_insert,
    remove: selection_remove,
    clone: selection_clone,
    datum: selection_datum,
    on: selection_on,
    dispatch: selection_dispatch
  };
  function select(selector) {
    return typeof selector === "string" ? new Selection([[document.querySelector(selector)]], [document.documentElement]) : new Selection([[selector]], root);
  }
  function sourceEvent() {
    var current = event,
      source;
    while (source = current.sourceEvent) current = source;
    return current;
  }
  function point(node, event) {
    var svg = node.ownerSVGElement || node;
    if (svg.createSVGPoint) {
      var point = svg.createSVGPoint();
      point.x = event.clientX, point.y = event.clientY;
      point = point.matrixTransform(node.getScreenCTM().inverse());
      return [point.x, point.y];
    }
    var rect = node.getBoundingClientRect();
    return [event.clientX - rect.left - node.clientLeft, event.clientY - rect.top - node.clientTop];
  }
  function mouse(node) {
    var event = sourceEvent();
    if (event.changedTouches) event = event.changedTouches[0];
    return point(node, event);
  }
  function touch(node, touches, identifier) {
    if (arguments.length < 3) identifier = touches, touches = sourceEvent().changedTouches;
    for (var i = 0, n = touches ? touches.length : 0, touch; i < n; ++i) {
      if ((touch = touches[i]).identifier === identifier) {
        return point(node, touch);
      }
    }
    return null;
  }
  function nopropagation() {
    event.stopImmediatePropagation();
  }
  function noevent() {
    event.preventDefault();
    event.stopImmediatePropagation();
  }
  function nodrag(view) {
    var root = view.document.documentElement,
      selection = select(view).on("dragstart.drag", noevent, true);
    if ("onselectstart" in root) {
      selection.on("selectstart.drag", noevent, true);
    } else {
      root.__noselect = root.style.MozUserSelect;
      root.style.MozUserSelect = "none";
    }
  }
  function yesdrag(view, noclick) {
    var root = view.document.documentElement,
      selection = select(view).on("dragstart.drag", null);
    if (noclick) {
      selection.on("click.drag", noevent, true);
      setTimeout(function () {
        selection.on("click.drag", null);
      }, 0);
    }
    if ("onselectstart" in root) {
      selection.on("selectstart.drag", null);
    } else {
      root.style.MozUserSelect = root.__noselect;
      delete root.__noselect;
    }
  }
  function constant$3(x) {
    return function () {
      return x;
    };
  }
  function DragEvent(target, type, subject, id, active, x, y, dx, dy, dispatch) {
    this.target = target;
    this.type = type;
    this.subject = subject;
    this.identifier = id;
    this.active = active;
    this.x = x;
    this.y = y;
    this.dx = dx;
    this.dy = dy;
    this._ = dispatch;
  }
  DragEvent.prototype.on = function () {
    var value = this._.on.apply(this._, arguments);
    return value === this._ ? this : value;
  };

  // Ignore right-click, since that should open the context menu.
  function defaultFilter() {
    return !event.ctrlKey && !event.button;
  }
  function defaultContainer() {
    return this.parentNode;
  }
  function defaultSubject(d) {
    return d == null ? {
      x: event.x,
      y: event.y
    } : d;
  }
  function defaultTouchable() {
    return navigator.maxTouchPoints || "ontouchstart" in this;
  }
  function drag() {
    var filter = defaultFilter,
      container = defaultContainer,
      subject = defaultSubject,
      touchable = defaultTouchable,
      gestures = {},
      listeners = dispatch("start", "drag", "end"),
      active = 0,
      mousedownx,
      mousedowny,
      mousemoving,
      touchending,
      clickDistance2 = 0;
    function drag(selection) {
      selection.on("mousedown.drag", mousedowned).filter(touchable).on("touchstart.drag", touchstarted).on("touchmove.drag", touchmoved).on("touchend.drag touchcancel.drag", touchended).style("touch-action", "none").style("-webkit-tap-highlight-color", "rgba(0,0,0,0)");
    }
    function mousedowned() {
      if (touchending || !filter.apply(this, arguments)) return;
      var gesture = beforestart("mouse", container.apply(this, arguments), mouse, this, arguments);
      if (!gesture) return;
      select(event.view).on("mousemove.drag", mousemoved, true).on("mouseup.drag", mouseupped, true);
      nodrag(event.view);
      nopropagation();
      mousemoving = false;
      mousedownx = event.clientX;
      mousedowny = event.clientY;
      gesture("start");
    }
    function mousemoved() {
      noevent();
      if (!mousemoving) {
        var dx = event.clientX - mousedownx,
          dy = event.clientY - mousedowny;
        mousemoving = dx * dx + dy * dy > clickDistance2;
      }
      gestures.mouse("drag");
    }
    function mouseupped() {
      select(event.view).on("mousemove.drag mouseup.drag", null);
      yesdrag(event.view, mousemoving);
      noevent();
      gestures.mouse("end");
    }
    function touchstarted() {
      if (!filter.apply(this, arguments)) return;
      var touches = event.changedTouches,
        c = container.apply(this, arguments),
        n = touches.length,
        i,
        gesture;
      for (i = 0; i < n; ++i) {
        if (gesture = beforestart(touches[i].identifier, c, touch, this, arguments)) {
          nopropagation();
          gesture("start");
        }
      }
    }
    function touchmoved() {
      var touches = event.changedTouches,
        n = touches.length,
        i,
        gesture;
      for (i = 0; i < n; ++i) {
        if (gesture = gestures[touches[i].identifier]) {
          noevent();
          gesture("drag");
        }
      }
    }
    function touchended() {
      var touches = event.changedTouches,
        n = touches.length,
        i,
        gesture;
      if (touchending) clearTimeout(touchending);
      touchending = setTimeout(function () {
        touchending = null;
      }, 500); // Ghost clicks are delayed!
      for (i = 0; i < n; ++i) {
        if (gesture = gestures[touches[i].identifier]) {
          nopropagation();
          gesture("end");
        }
      }
    }
    function beforestart(id, container, point, that, args) {
      var p = point(container, id),
        s,
        dx,
        dy,
        sublisteners = listeners.copy();
      if (!customEvent(new DragEvent(drag, "beforestart", s, id, active, p[0], p[1], 0, 0, sublisteners), function () {
        if ((event.subject = s = subject.apply(that, args)) == null) return false;
        dx = s.x - p[0] || 0;
        dy = s.y - p[1] || 0;
        return true;
      })) return;
      return function gesture(type) {
        var p0 = p,
          n;
        switch (type) {
          case "start":
            gestures[id] = gesture, n = active++;
            break;
          case "end":
            delete gestures[id], --active;
          // nobreak
          case "drag":
            p = point(container, id), n = active;
            break;
        }
        customEvent(new DragEvent(drag, type, s, id, n, p[0] + dx, p[1] + dy, p[0] - p0[0], p[1] - p0[1], sublisteners), sublisteners.apply, sublisteners, [type, that, args]);
      };
    }
    drag.filter = function (_) {
      return arguments.length ? (filter = typeof _ === "function" ? _ : constant$3(!!_), drag) : filter;
    };
    drag.container = function (_) {
      return arguments.length ? (container = typeof _ === "function" ? _ : constant$3(_), drag) : container;
    };
    drag.subject = function (_) {
      return arguments.length ? (subject = typeof _ === "function" ? _ : constant$3(_), drag) : subject;
    };
    drag.touchable = function (_) {
      return arguments.length ? (touchable = typeof _ === "function" ? _ : constant$3(!!_), drag) : touchable;
    };
    drag.on = function () {
      var value = listeners.on.apply(listeners, arguments);
      return value === listeners ? drag : value;
    };
    drag.clickDistance = function (_) {
      return arguments.length ? (clickDistance2 = (_ = +_) * _, drag) : Math.sqrt(clickDistance2);
    };
    return drag;
  }

  // Copyright 2018 The Distill Template Authors

  var T$a = Template('d-slider', "\n<style>\n  :host {\n    position: relative;\n    display: inline-block;\n  }\n\n  :host(:focus) {\n    outline: none;\n  }\n\n  .background {\n    padding: 9px 0;\n    color: white;\n    position: relative;\n  }\n\n  .track {\n    height: 3px;\n    width: 100%;\n    border-radius: 2px;\n    background-color: hsla(0, 0%, 0%, 0.2);\n  }\n\n  .track-fill {\n    position: absolute;\n    top: 9px;\n    height: 3px;\n    border-radius: 4px;\n    background-color: hsl(24, 100%, 50%);\n  }\n\n  .knob-container {\n    position: absolute;\n    top: 10px;\n  }\n\n  .knob {\n    position: absolute;\n    top: -6px;\n    left: -6px;\n    width: 13px;\n    height: 13px;\n    background-color: hsl(24, 100%, 50%);\n    border-radius: 50%;\n    transition-property: transform;\n    transition-duration: 0.18s;\n    transition-timing-function: ease;\n  }\n  .mousedown .knob {\n    transform: scale(1.5);\n  }\n\n  .knob-highlight {\n    position: absolute;\n    top: -6px;\n    left: -6px;\n    width: 13px;\n    height: 13px;\n    background-color: hsla(0, 0%, 0%, 0.1);\n    border-radius: 50%;\n    transition-property: transform;\n    transition-duration: 0.18s;\n    transition-timing-function: ease;\n  }\n\n  .focus .knob-highlight {\n    transform: scale(2);\n  }\n\n  .ticks {\n    position: absolute;\n    top: 16px;\n    height: 4px;\n    width: 100%;\n    z-index: -1;\n  }\n\n  .ticks .tick {\n    position: absolute;\n    height: 100%;\n    border-left: 1px solid hsla(0, 0%, 0%, 0.2);\n  }\n\n</style>\n\n  <div class='background'>\n    <div class='track'></div>\n    <div class='track-fill'></div>\n    <div class='knob-container'>\n      <div class='knob-highlight'></div>\n      <div class='knob'></div>\n    </div>\n    <div class='ticks'></div>\n  </div>\n");

  // ARIA
  // If the slider has a visible label, it is referenced by aria-labelledby on the slider element. Otherwise, the slider element has a label provided by aria-label.
  // If the slider is vertically oriented, it has aria-orientation set to vertical. The default value of aria-orientation for a slider is horizontal.

  var keyCodes = {
    left: 37,
    up: 38,
    right: 39,
    down: 40,
    pageUp: 33,
    pageDown: 34,
    end: 35,
    home: 36
  };
  var Slider = /*#__PURE__*/function (_T$a) {
    function Slider() {
      _classCallCheck(this, Slider);
      return _callSuper(this, Slider, arguments);
    }
    _inherits(Slider, _T$a);
    return _createClass(Slider, [{
      key: "connectedCallback",
      value: function connectedCallback() {
        var _this19 = this;
        this.connected = true;
        this.setAttribute('role', 'slider');
        // Makes the element tab-able.
        if (!this.hasAttribute('tabindex')) {
          this.setAttribute('tabindex', 0);
        }

        // Keeps track of keyboard vs. mouse interactions for focus rings
        this.mouseEvent = false;

        // Handles to shadow DOM elements
        this.knob = this.root.querySelector('.knob-container');
        this.background = this.root.querySelector('.background');
        this.trackFill = this.root.querySelector('.track-fill');
        this.track = this.root.querySelector('.track');

        // Default values for attributes
        this.min = this.min ? this.min : 0;
        this.max = this.max ? this.max : 100;
        this.scale = linear$1().domain([this.min, this.max]).range([0, 1]).clamp(true);
        this.origin = this.origin !== undefined ? this.origin : this.min;
        this.step = this.step ? this.step : 1;
        this.update(this.value ? this.value : 0);
        this.ticks = this.ticks ? this.ticks : false;
        this.renderTicks();
        this.drag = drag().container(this.background).on('start', function () {
          _this19.mouseEvent = true;
          _this19.background.classList.add('mousedown');
          _this19.changeValue = _this19.value;
          _this19.dragUpdate();
        }).on('drag', function () {
          _this19.dragUpdate();
        }).on('end', function () {
          _this19.mouseEvent = false;
          _this19.background.classList.remove('mousedown');
          _this19.dragUpdate();
          if (_this19.changeValue !== _this19.value) _this19.dispatchChange();
          _this19.changeValue = _this19.value;
        });
        this.drag(select(this.background));
        this.addEventListener('focusin', function () {
          if (!_this19.mouseEvent) {
            _this19.background.classList.add('focus');
          }
        });
        this.addEventListener('focusout', function () {
          _this19.background.classList.remove('focus');
        });
        this.addEventListener('keydown', this.onKeyDown);
      }
    }, {
      key: "attributeChangedCallback",
      value: function attributeChangedCallback(attr, oldValue, newValue) {
        if (isNaN(newValue) || newValue === undefined || newValue === null) return;
        if (attr == 'min') {
          this.min = +newValue;
          this.setAttribute('aria-valuemin', this.min);
        }
        if (attr == 'max') {
          this.max = +newValue;
          this.setAttribute('aria-valuemax', this.max);
        }
        if (attr == 'value') {
          this.update(+newValue);
        }
        if (attr == 'origin') {
          this.origin = +newValue;
          // this.update(this.value);
        }
        if (attr == 'step') {
          if (newValue > 0) {
            this.step = +newValue;
          }
        }
        if (attr == 'ticks') {
          this.ticks = newValue === '' ? true : newValue;
        }
      }
    }, {
      key: "onKeyDown",
      value: function onKeyDown(event) {
        this.changeValue = this.value;
        var stopPropagation = false;
        switch (event.keyCode) {
          case keyCodes.left:
          case keyCodes.down:
            this.update(this.value - this.step);
            stopPropagation = true;
            break;
          case keyCodes.right:
          case keyCodes.up:
            this.update(this.value + this.step);
            stopPropagation = true;
            break;
          case keyCodes.pageUp:
            this.update(this.value + this.step * 10);
            stopPropagation = true;
            break;
          case keyCodes.pageDown:
            this.update(this.value + this.step * 10);
            stopPropagation = true;
            break;
          case keyCodes.home:
            this.update(this.min);
            stopPropagation = true;
            break;
          case keyCodes.end:
            this.update(this.max);
            stopPropagation = true;
            break;
        }
        if (stopPropagation) {
          this.background.classList.add('focus');
          event.preventDefault();
          event.stopPropagation();
          if (this.changeValue !== this.value) this.dispatchChange();
        }
      }
    }, {
      key: "validateValueRange",
      value: function validateValueRange(min, max, value) {
        return Math.max(Math.min(max, value), min);
      }
    }, {
      key: "quantizeValue",
      value: function quantizeValue(value, step) {
        return Math.round(value / step) * step;
      }
    }, {
      key: "dragUpdate",
      value: function dragUpdate() {
        var bbox = this.background.getBoundingClientRect();
        var x = event.x;
        var width = bbox.width;
        this.update(this.scale.invert(x / width));
      }
    }, {
      key: "update",
      value: function update(value) {
        var v = value;
        if (this.step !== 'any') {
          v = this.quantizeValue(value, this.step);
        }
        v = this.validateValueRange(this.min, this.max, v);
        if (this.connected) {
          this.knob.style.left = this.scale(v) * 100 + '%';
          this.trackFill.style.width = this.scale(this.min + Math.abs(v - this.origin)) * 100 + '%';
          this.trackFill.style.left = this.scale(Math.min(v, this.origin)) * 100 + '%';
        }
        if (this.value !== v) {
          this.value = v;
          this.setAttribute('aria-valuenow', this.value);
          this.dispatchInput();
        }
      }

      // Dispatches only on a committed change (basically only on mouseup).
    }, {
      key: "dispatchChange",
      value: function dispatchChange() {
        var e = new Event('change');
        this.dispatchEvent(e, {});
      }

      // Dispatches on each value change.
    }, {
      key: "dispatchInput",
      value: function dispatchInput() {
        var e = new Event('input');
        this.dispatchEvent(e, {});
      }
    }, {
      key: "renderTicks",
      value: function renderTicks() {
        var _this20 = this;
        var ticksContainer = this.root.querySelector('.ticks');
        if (this.ticks !== false) {
          var tickData = [];
          if (this.ticks > 0) {
            tickData = this.scale.ticks(this.ticks);
          } else if (this.step === 'any') {
            tickData = this.scale.ticks();
          } else {
            tickData = range(this.min, this.max + 1e-6, this.step);
          }
          tickData.forEach(function (d) {
            var tick = document.createElement('div');
            tick.classList.add('tick');
            tick.style.left = _this20.scale(d) * 100 + '%';
            ticksContainer.appendChild(tick);
          });
        } else {
          ticksContainer.style.display = 'none';
        }
      }
    }], [{
      key: "observedAttributes",
      get: function get() {
        return ['min', 'max', 'value', 'step', 'ticks', 'origin', 'tickValues', 'tickLabels'];
      }
    }]);
  }(T$a(HTMLElement));
  var logo = "<svg viewBox=\"-607 419 64 64\">\n  <path d=\"M-573.4,478.9c-8,0-14.6-6.4-14.6-14.5s14.6-25.9,14.6-40.8c0,14.9,14.6,32.8,14.6,40.8S-565.4,478.9-573.4,478.9z\"/>\n</svg>\n";
  var headerTemplate = "\n<style>\ndistill-header {\n  position: relative;\n  height: 60px;\n  background-color: hsl(200, 60%, 15%);\n  width: 100%;\n  box-sizing: border-box;\n  z-index: 2;\n  color: rgba(0, 0, 0, 0.8);\n  border-bottom: 1px solid rgba(0, 0, 0, 0.08);\n  box-shadow: 0 1px 6px rgba(0, 0, 0, 0.05);\n}\ndistill-header .content {\n  height: 70px;\n  grid-column: page;\n}\ndistill-header a {\n  font-size: 16px;\n  height: 60px;\n  line-height: 60px;\n  text-decoration: none;\n  color: rgba(255, 255, 255, 0.8);\n  padding: 22px 0;\n}\ndistill-header a:hover {\n  color: rgba(255, 255, 255, 1);\n}\ndistill-header svg {\n  width: 24px;\n  position: relative;\n  top: 4px;\n  margin-right: 2px;\n}\n@media(min-width: 1080px) {\n  distill-header {\n    height: 70px;\n  }\n  distill-header a {\n    height: 70px;\n    line-height: 70px;\n    padding: 28px 0;\n  }\n  distill-header .logo {\n  }\n}\ndistill-header svg path {\n  fill: none;\n  stroke: rgba(255, 255, 255, 0.8);\n  stroke-width: 3px;\n}\ndistill-header .logo {\n  font-size: 17px;\n  font-weight: 200;\n}\ndistill-header .nav {\n  float: right;\n  font-weight: 300;\n}\ndistill-header .nav a {\n  font-size: 12px;\n  margin-left: 24px;\n  text-transform: uppercase;\n}\n</style>\n<div class=\"content\">\n  <a href=\"/\" class=\"logo\">\n    ".concat(logo, "\n    Distill\n  </a>\n  <nav class=\"nav\">\n    <a href=\"/about/\">About</a>\n    <a href=\"/prize/\">Prize</a>\n    <a href=\"/journal/\">Submit</a>\n  </nav>\n</div>\n");

  // Copyright 2018 The Distill Template Authors

  var T$b = Template('distill-header', headerTemplate, false);
  var DistillHeader = /*#__PURE__*/function (_T$b) {
    function DistillHeader() {
      _classCallCheck(this, DistillHeader);
      return _callSuper(this, DistillHeader, arguments);
    }
    _inherits(DistillHeader, _T$b);
    return _createClass(DistillHeader);
  }(T$b(HTMLElement)); // Copyright 2018 The Distill Template Authors
  var styles$2 = "\n<style>\n  distill-appendix {\n    contain: layout style;\n  }\n\n  distill-appendix .citation {\n    font-size: 11px;\n    line-height: 15px;\n    border-left: 1px solid rgba(0, 0, 0, 0.1);\n    padding-left: 18px;\n    border: 1px solid rgba(0,0,0,0.1);\n    background: rgba(0, 0, 0, 0.02);\n    padding: 10px 18px;\n    border-radius: 3px;\n    color: rgba(150, 150, 150, 1);\n    overflow: hidden;\n    margin-top: -12px;\n    white-space: pre-wrap;\n    word-wrap: break-word;\n  }\n\n  distill-appendix > * {\n    grid-column: text;\n  }\n</style>\n";
  function appendixTemplate(frontMatter) {
    var html = styles$2;
    if (typeof frontMatter.githubUrl !== 'undefined') {
      html += "\n    <h3 id=\"updates-and-corrections\">Updates and Corrections</h3>\n    <p>";
      if (frontMatter.githubCompareUpdatesUrl) {
        html += "<a href=\"".concat(frontMatter.githubCompareUpdatesUrl, "\">View all changes</a> to this article since it was first published.");
      }
      html += "\n    If you see mistakes or want to suggest changes, please <a href=\"".concat(frontMatter.githubUrl + '/issues/new', "\">create an issue on GitHub</a>. </p>\n    ");
    }
    var journal = frontMatter.journal;
    if (typeof journal !== 'undefined' && journal.title === 'Distill') {
      html += "\n    <h3 id=\"reuse\">Reuse</h3>\n    <p>Diagrams and text are licensed under Creative Commons Attribution <a href=\"https://creativecommons.org/licenses/by/4.0/\">CC-BY 4.0</a> with the <a class=\"github\" href=\"".concat(frontMatter.githubUrl, "\">source available on GitHub</a>, unless noted otherwise. The figures that have been reused from other sources don\u2019t fall under this license and can be recognized by a note in their caption: \u201CFigure from \u2026\u201D.</p>\n    ");
    }
    if (typeof frontMatter.publishedDate !== 'undefined') {
      html += "\n    <h3 id=\"citation\">Citation</h3>\n    <p>For attribution in academic contexts, please cite this work as</p>\n    <pre class=\"citation short\">".concat(frontMatter.concatenatedAuthors, ", \"").concat(frontMatter.title, "\", Distill, ").concat(frontMatter.publishedYear, ".</pre>\n    <p>BibTeX citation</p>\n    <pre class=\"citation long\">").concat(serializeFrontmatterToBibtex(frontMatter), "</pre>\n    ");
    }
    return html;
  }
  var DistillAppendix = /*#__PURE__*/function (_HTMLElement9) {
    function DistillAppendix() {
      _classCallCheck(this, DistillAppendix);
      return _callSuper(this, DistillAppendix, arguments);
    }
    _inherits(DistillAppendix, _HTMLElement9);
    return _createClass(DistillAppendix, [{
      key: "frontMatter",
      set: function set(frontMatter) {
        this.innerHTML = appendixTemplate(frontMatter);
      }
    }], [{
      key: "is",
      get: function get() {
        return 'distill-appendix';
      }
    }]);
  }(/*#__PURE__*/_wrapNativeSuper(HTMLElement));
  var footerTemplate = "\n<style>\n\n:host {\n  color: rgba(255, 255, 255, 0.5);\n  font-weight: 300;\n  padding: 2rem 0;\n  border-top: 1px solid rgba(0, 0, 0, 0.1);\n  background-color: hsl(180, 5%, 15%); /*hsl(200, 60%, 15%);*/\n  text-align: left;\n  contain: content;\n}\n\n.footer-container .logo svg {\n  width: 24px;\n  position: relative;\n  top: 4px;\n  margin-right: 2px;\n}\n\n.footer-container .logo svg path {\n  fill: none;\n  stroke: rgba(255, 255, 255, 0.8);\n  stroke-width: 3px;\n}\n\n.footer-container .logo {\n  font-size: 17px;\n  font-weight: 200;\n  color: rgba(255, 255, 255, 0.8);\n  text-decoration: none;\n  margin-right: 6px;\n}\n\n.footer-container {\n  grid-column: text;\n}\n\n.footer-container .nav {\n  font-size: 0.9em;\n  margin-top: 1.5em;\n}\n\n.footer-container .nav a {\n  color: rgba(255, 255, 255, 0.8);\n  margin-right: 6px;\n  text-decoration: none;\n}\n\n</style>\n\n<div class='footer-container'>\n\n  <a href=\"/\" class=\"logo\">\n    ".concat(logo, "\n    Distill\n  </a> is dedicated to clear explanations of machine learning\n\n  <div class=\"nav\">\n    <a href=\"https://distill.pub/about/\">About</a>\n    <a href=\"https://distill.pub/journal/\">Submit</a>\n    <a href=\"https://distill.pub/prize/\">Prize</a>\n    <a href=\"https://distill.pub/archive/\">Archive</a>\n    <a href=\"https://distill.pub/rss.xml\">RSS</a>\n    <a href=\"https://github.com/distillpub\">GitHub</a>\n    <a href=\"https://twitter.com/distillpub\">Twitter</a>\n    &nbsp;&nbsp;&nbsp;&nbsp; ISSN 2476-0757\n  </div>\n\n</div>\n\n");

  // Copyright 2018 The Distill Template Authors

  var T$c = Template('distill-footer', footerTemplate);
  var DistillFooter = /*#__PURE__*/function (_T$c) {
    function DistillFooter() {
      _classCallCheck(this, DistillFooter);
      return _callSuper(this, DistillFooter, arguments);
    }
    _inherits(DistillFooter, _T$c);
    return _createClass(DistillFooter);
  }(T$c(HTMLElement)); // Copyright 2018 The Distill Template Authors
  var templateIsLoading = false;
  var runlevel = 0;
  var initialize = function initialize() {
    if (window.distill.runlevel < 1) {
      throw new Error("Insufficient Runlevel for Distill Template!");
    }

    /* 1. Flag that we're being loaded */
    if ("distill" in window && window.distill.templateIsLoading) {
      throw new Error("Runlevel 1: Distill Template is getting loaded more than once, aborting!");
    } else {
      window.distill.templateIsLoading = true;
      console.debug("Runlevel 1: Distill Template has started loading.");
    }

    /* 2. Add styles if they weren't added during prerendering */
    makeStyleTag(document);
    console.debug("Runlevel 1: Static Distill styles have been added.");
    console.debug("Runlevel 1->2.");
    window.distill.runlevel += 1;

    /* 3. Register Controller listener functions */
    /* Needs to happen before components to their connected callbacks have a controller to talk to. */
    for (var _i2 = 0, _Object$entries2 = Object.entries(Controller.listeners); _i2 < _Object$entries2.length; _i2++) {
      var _Object$entries2$_i = _slicedToArray(_Object$entries2[_i2], 2),
        functionName = _Object$entries2$_i[0],
        callback = _Object$entries2$_i[1];
      if (typeof callback === "function") {
        document.addEventListener(functionName, callback);
      } else {
        console.error("Runlevel 2: Controller listeners need to be functions!");
      }
    }
    console.debug("Runlevel 2: We can now listen to controller events.");
    console.debug("Runlevel 2->3.");
    window.distill.runlevel += 1;

    /* 4. Register components */
    var components = [Abstract, Appendix, Article, Bibliography, Byline, Cite, CitationList, Code, Footnote, FootnoteList, FrontMatter$1, HoverBox, Title, DMath, References, TOC, Figure, Slider, Interstitial];
    var distillComponents = [DistillHeader, DistillAppendix, DistillFooter];
    if (window.distill.runlevel < 2) {
      throw new Error("Insufficient Runlevel for adding custom elements!");
    }
    var allComponents = components.concat(distillComponents);
    var _iterator21 = _createForOfIteratorHelper(allComponents),
      _step21;
    try {
      for (_iterator21.s(); !(_step21 = _iterator21.n()).done;) {
        var component = _step21.value;
        console.debug("Runlevel 2: Registering custom element: " + component.is);
        customElements.define(component.is, component);
      }
    } catch (err) {
      _iterator21.e(err);
    } finally {
      _iterator21.f();
    }
    console.debug("Runlevel 3: Distill Template finished registering custom elements.");
    console.debug("Runlevel 3->4.");
    window.distill.runlevel += 1;

    // If template was added after DOMContentLoaded we may have missed that event.
    // Controller will check for that case, so trigger the event explicitly:
    if (domContentLoaded()) {
      Controller.listeners.DOMContentLoaded();
    }
    console.debug("Runlevel 4: Distill Template initialisation complete.");
    window.distill.templateIsLoading = false;
    window.distill.templateHasLoaded = true;
  };
  window.distill = {
    runlevel: runlevel,
    initialize: initialize,
    templateIsLoading: templateIsLoading
  };

  /* 0. Check browser feature support; synchronously polyfill if needed */
  if (Polyfills.browserSupportsAllFeatures()) {
    console.debug("Runlevel 0: No need for polyfills.");
    console.debug("Runlevel 0->1.");
    window.distill.runlevel += 1;
    window.distill.initialize();
  } else {
    console.debug("Runlevel 0: Distill Template is loading polyfills.");
    Polyfills.load(window.distill.initialize);
  }
});

/***/ })

/******/ 	});
/************************************************************************/
/******/ 	// The module cache
/******/ 	var __webpack_module_cache__ = {};
/******/ 	
/******/ 	// The require function
/******/ 	function __webpack_require__(moduleId) {
/******/ 		// Check if module is in cache
/******/ 		var cachedModule = __webpack_module_cache__[moduleId];
/******/ 		if (cachedModule !== undefined) {
/******/ 			return cachedModule.exports;
/******/ 		}
/******/ 		// Create a new module (and put it into the cache)
/******/ 		var module = __webpack_module_cache__[moduleId] = {
/******/ 			// no module.id needed
/******/ 			// no module.loaded needed
/******/ 			exports: {}
/******/ 		};
/******/ 	
/******/ 		// Execute the module function
/******/ 		__webpack_modules__[moduleId](module, module.exports, __webpack_require__);
/******/ 	
/******/ 		// Return the exports of the module
/******/ 		return module.exports;
/******/ 	}
/******/ 	
/************************************************************************/
/******/ 	/* webpack/runtime/global */
/******/ 	(() => {
/******/ 		__webpack_require__.g = (function() {
/******/ 			if (typeof globalThis === 'object') return globalThis;
/******/ 			try {
/******/ 				return this || new Function('return this')();
/******/ 			} catch (e) {
/******/ 				if (typeof window === 'object') return window;
/******/ 			}
/******/ 		})();
/******/ 	})();
/******/ 	
/************************************************************************/
/******/ 	
/******/ 	// startup
/******/ 	// Load entry module and return exports
/******/ 	// This entry module used 'module' so it can't be inlined
/******/ 	var __webpack_exports__ = __webpack_require__(792);
/******/ 	
/******/ })()
;
//# sourceMappingURL=distill.bundle.js.map