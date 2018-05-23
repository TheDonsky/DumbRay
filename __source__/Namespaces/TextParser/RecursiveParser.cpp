#include "RecursiveParser.h"
#include "Error.h"
#include <iostream>


namespace Parser {
#define SRC "RecursiveParser.cpp" 


	namespace {
		static void addAllPrefixes(ByteSet &set, const std::string &text, char *buffer) {
			char *buff = buffer;
			if (buff == NULL) buff = new char[text.length() + 1];
			if (buff == NULL) Error::fatal("Allocation failed", SRC, "addAllPrefixes", Error::ALLOCATION_FALURE);
			for (unsigned int i = 0; i <= text.length(); i++) {
				buff[i] = '\0';
				set.addString(buff);
				buff[i] = text[i];
			}
			if (buff != buffer) delete[] buff;
		}

		template<typename Type>
		inline static void addToKeywordMap(ByteMap<Type> &map, ByteSet &preffs, const std::string &key, const Type &value, char *buffer) {
			addAllPrefixes(preffs, key, buffer);
			map.addString(value, key.c_str());
		}

		static size_t loadC_str(char *cursor, const char *end, const char *text) {
			size_t i = 0;
			while (text[i] != '\0') {
				if (cursor >= end) Error::fatal("Buffer overflow", SRC, "loadC_str", Error::INVALID_POINTER);
				else {
					(*cursor) = text[i];
					cursor++;
				}
				i++;
			}
			return i;
		}
		/*
		static void load(char *cursor, const char *end, const RecursiveParser::Text &text) {
			loadC_str(cursor, end, text.c_str());
		}
		*/
		static size_t loadStringC_str(char *cursor, const char *end, const char *text) {
			size_t length = loadC_str(cursor, end, text);
			if ((cursor + length) >= end) Error::fatal("Buffer overflow", SRC, "loadString", Error::INVALID_POINTER);
			else (*(cursor + length)) = '\0';
			return length;
		}
		static void loadString(char *cursor, const char *end, const RecursiveParser::Text &text) {
			loadStringC_str(cursor, end, text.c_str());
		}
		static size_t loadBackC_str(char *&cursor, char *buffStart, const char *text, size_t length) {
			if (((size_t)(cursor - buffStart)) < length) Error::fatal("Buffer underflow", SRC, "loadBack", Error::INVALID_POINTER);
			else {
				cursor -= length;
				loadC_str(cursor, cursor + length, text);
			}
			return length;
		}
		static size_t loadBackC_str(char *&cursor, char *buffStart, const char *text) {
			const char *t = text;
			while ((*t) != '\0') t++;
			return loadBackC_str(cursor, buffStart, text, (t - text));
		}
		static void loadBack(char *&cursor, char *buffStart, const RecursiveParser::Text &text) {
			loadBackC_str(cursor, buffStart, text.c_str(), text.length());
		}
		static void addToken(std::vector<RecursiveParser::Token> &tokens, RecursiveParser::Token &token) {
			if (token.text.length() > 0 || token.entry.length() > 0 || token.subTokens.size() > 0 || token.exit.length() > 0)
				tokens.push_back(token);
		}
		enum ExitType {
			TOKEN_ACTION_NONE = 0,
			TOKEN_EXIT = 1,
			BREAK_TOKEN = 2,
			PRESERVE = 4
		};

		static void unescapedPrint(const std::string &string, std::ostream &stream) {
			for (size_t i = 0; i < string.length(); i++) {
				if (string[i] == '\n') stream << "\\n";
				else if (string[i] == '\r') stream << "\\r";
				else stream << string[i];
			}
		}

		static void dumpRecursiveTokens(const std::vector<RecursiveParser::Token> &tokens, const std::string &prefix, std::ostream &stream);
		static void dumpRecursiveToken(const RecursiveParser::Token &token, const std::string &prefix, std::ostream &stream) {
			stream << prefix << "[" << token.text << "]";
			if (token.entry.length() > 0) {
				stream << " <";
				unescapedPrint(token.entry, stream);
				stream << "> ";
			}
			if (token.subTokens.size() > 0) {
				stream << std::endl;
				dumpRecursiveTokens(token.subTokens, prefix + "    ", stream);
				stream << prefix;
			}
			if (token.exit.length() > 0) {
				stream << "<";
				unescapedPrint(token.exit, stream);
				stream << ">";
			}
			stream << std::endl;
		}
		static void dumpRecursiveTokens(const std::vector<RecursiveParser::Token> &tokens, const std::string &prefix, std::ostream &stream) {
			for (size_t i = 0; i < tokens.size(); i++)
				dumpRecursiveToken(tokens[i], prefix, stream);
		}
	}

	void RecursiveParser::Token::dump(std::ostream &stream)const {
		dumpRecursiveToken((*this), "", stream);
	}
	void RecursiveParser::Token::dump(const std::vector<Token> &tokens, std::ostream &stream) {
		dumpRecursiveTokens(tokens, "", stream);
	}

	RecursiveParser::RecursiveParser() {
		preParserAddOn = "";
		postParserAddOn = "";
		maxEscapeScale = 1;
		maxPreParserAddOnSize = 0;
		maxPostParserAddOnSize = 0;
	}
	RecursiveParser::~RecursiveParser() {
	}


	void RecursiveParser::addEscapeSequence(const Text &sequence, const Text &replacement, EscapeType type) {
		EscapeValue value;
		value.replacement = replacement;
		value.type = type;
		addToKeywordMap(escapeSequences.map, escapeSequences.prefixes, sequence, value, NULL);
		size_t escapeScale = (((replacement.length() - 1) / sequence.length()) + 1);
		if (escapeScale < 1) escapeScale = 1;
		if (maxEscapeScale < escapeScale) maxEscapeScale = escapeScale;
	}
	void RecursiveParser::addIgnoredSequence(const Text &sequence) {
		addEscapeSequence(sequence, "", ESCAPE_REPLACE);
	}
	void RecursiveParser::addEntryPoint(const Keyword &key, bool preserve) {
		EntryPoint point;
		point.key = key;
		point.preserve = preserve;
		entryPoints.push_back(point);
	}
	void RecursiveParser::addExitPoint(const Keyword &key, bool preserve) {
		addExitPointOrTokenBreaker(key, TOKEN_EXIT | (preserve ? PRESERVE : TOKEN_ACTION_NONE));
	}
	void RecursiveParser::addDelimiter(const Keyword &key) {
		addExitPointOrTokenBreaker(key, BREAK_TOKEN);
	}
	void RecursiveParser::setPreParseAddOn(const Text &text) {
		preParserAddOn = text;
		if (maxPreParserAddOnSize < text.length()) maxPreParserAddOnSize = text.length();
	}
	void RecursiveParser::setPostParseAddOn(const Text &text) {
		postParserAddOn = text;
		if (maxPostParserAddOnSize < text.length()) maxPostParserAddOnSize = text.length();
	}
	void RecursiveParser::addSubParser(const RecursiveParser *parser) {
		if (parser == NULL) Error::fatal("NULL pointer passed to the function", SRC, "RecursiveParser::addSubParser", Error::NULL_POINTER);
		size_t maxEntrySize = 0;
		for (size_t i = 0; i < parser->entryPoints.size(); i++)
			if (maxEntrySize < parser->entryPoints[i].key.length())
				maxEntrySize = parser->entryPoints[i].key.length();
		char *buffer = new char[maxEntrySize + 1];
		if (buffer == NULL) Error::fatal("Buffer Allocation failed", SRC, "RecursiveParser::addSubParser", Error::ALLOCATION_FALURE);
		for (unsigned int i = 0; i < parser->entryPoints.size(); i++) {
			SubParser subParser;
			subParser.parser = parser;
			subParser.preserveEntryPoint = parser->entryPoints[i].preserve;
			addToKeywordMap(subParsers.map, subParsers.prefixes, parser->entryPoints[i].key, subParser, buffer);
		}
		if (maxEscapeScale < parser->maxEscapeScale) maxEscapeScale = parser->maxEscapeScale;
		if (maxPreParserAddOnSize < parser->maxPreParserAddOnSize) maxPreParserAddOnSize = parser->maxPreParserAddOnSize;
		if (maxPostParserAddOnSize < parser->maxPostParserAddOnSize) maxPostParserAddOnSize = parser->maxPostParserAddOnSize;
		delete[] buffer;
	}


	std::vector<RecursiveParser::Token> RecursiveParser::parse(const Text &input)const {
		std::vector<Token> rv;
		parse(input, rv);
		return rv;
	}


	void RecursiveParser::parse(const Text &input, std::vector<Token> &output)const {
		size_t padding = (maxEscapeScale * (maxPreParserAddOnSize + maxPostParserAddOnSize + 8));
		size_t bufferSize = ((maxEscapeScale * input.length()) + (2 * padding) + 1);
		char *buffer = new char[bufferSize];
		if (buffer == NULL) Error::fatal("Buffer Allocation failed", SRC, "RecursiveParser::parse(input, output)", Error::ALLOCATION_FALURE);
		else {
			char *tmpBuffer = new char[bufferSize * 2];
			if (tmpBuffer == NULL) {
				delete[] buffer;
				Error::fatal("Failed to allocate tmp buffer", SRC, "RecursiveParser::parse(input, output)", Error::ALLOCATION_FALURE);
			}
			else {
				char *cursor = (buffer + padding);
				loadString(cursor, (buffer + bufferSize), input);
				loadBack(cursor, buffer, preParserAddOn);
				char *escapedBuffer = (tmpBuffer + bufferSize);
				char *tempBuffer = tmpBuffer;
				std::string exitPnt;
				parse(output, cursor, buffer, tempBuffer, escapedBuffer, tmpBuffer + bufferSize, escapedBuffer + bufferSize, exitPnt);
				delete[] buffer;
				delete[] tmpBuffer;
			}
		}
	}




	void RecursiveParser::addExitPointOrTokenBreaker(const Keyword &key, const ByteSet::Flags &flags) {
		addAllPrefixes(exitPointPreffixes, key, NULL);
		ByteSet::NodeId id = exitPoints.addString(key.c_str());
		if (BYTE_SET_NOT_A_NODE(id)) Error::fatal("Could not add entry in exitPoints", SRC, "RecursiveParser::addExitPoint", Error::VALUE_ERROR);
		exitPoints.flags(id) = flags;
	}

	void RecursiveParser::parse(std::vector<Token> &output, char *&cursor, char *buffStart, char *&tmpBuffer, char *&escapedBuffer, const char *tmpEnd, const char *escEnd, std::string &exitPnt)const {
		const char *noEscape = buffStart;
		while ((*cursor) != '\0') {
			char *tmpCursor = tmpBuffer;
			char *escCursor = escapedBuffer;
			(*tmpCursor) = '\0';
			(*escCursor) = '\0';
			Token token;
			while (true) {
				//bool inc = true;
				if (tmpCursor >= tmpEnd)
					Error::fatal("Buffer overflow", SRC, "RecursiveParser::parse(output, cursor, tmpBuffer, tmpEnd)", Error::INVALID_POINTER);
				else {
					(*tmpCursor) = '\0';
					ByteSet::NodeId escapeNode = escapeSequences.map.findString(tmpBuffer);
					bool escapedChanged = false;
					if (!BYTE_SET_NOT_A_NODE(escapeNode)) {
						const EscapeValue &escapeValue = escapeSequences.map.value(escapeNode);
						if (escapeValue.type == ESCAPE_RECURSIVE)
							loadBack(cursor, buffStart, escapeValue.replacement);
						else if (escapeValue.type == ESCAPE_REPLACE) {
							if (cursor >= noEscape) {
								loadString(escCursor, escEnd, escapeValue.replacement);
								escCursor += escapeValue.replacement.length();
							}
							else {
								size_t tmpBufferLength = loadStringC_str(escCursor, escEnd, tmpBuffer);
								escCursor += tmpBufferLength;
							}
							escapedChanged = true;
						}
						else if (escapeValue.type == ESCAPE_DIRECT_TO_TEXT) {
							if (escCursor == escapedBuffer)
								token.text += escapeValue.replacement;
							else {
								loadBack(cursor, buffStart, tmpBuffer);
								escapedChanged = true;
							}
						}
						tmpCursor = tmpBuffer;
						(*tmpCursor) = '\0';
					}
					else if (((*cursor) == '\0') || BYTE_SET_NOT_A_NODE(escapeSequences.prefixes.findString(tmpBuffer))) {
						if ((*cursor) == '\0') {
							tmpCursor = tmpBuffer;
							size_t tmpLen = loadStringC_str(escCursor, escEnd, tmpBuffer);
							escCursor += tmpLen;
						}
						else while (tmpCursor != tmpBuffer) {
							(*escCursor) = (*tmpBuffer);
							escCursor++;
							if (escCursor >= escEnd)
								Error::fatal("Buffer overflow", SRC, "RecursiveParser::parse (private)", Error::INVALID_POINTER);
							(*escCursor) = '\0';
							tmpBuffer++;
							//escapedChanged = true;
							if (!BYTE_SET_NOT_A_NODE(escapeSequences.prefixes.findString(tmpBuffer))) {
								/*size_t delta = */ loadBackC_str(cursor, buffStart, tmpBuffer);
								tmpCursor = tmpBuffer;
								//escapedChanged = false;
								break;
							}
						}
						if ((*escapedBuffer) != '\0') escapedChanged = true;
						//if (escCursor > escapedBuffer) escapedChanged = true;
					}
					bool tokenEnded = false;
					if (escapedChanged) {
						// If we got here, we have something in escaped buffer and tmpBuffer is empty;
						const char *esc = escapedBuffer;
						char *tmp = tmpBuffer;
						char *tmpStart = tmp;
						while (true) {
							(*tmp) = '\0';
							ByteSet::NodeId entryId = subParsers.map.findString(tmpStart);
							if (!BYTE_SET_NOT_A_NODE(entryId)) {
								// Here we clean things up and transfer control to the sub-parser;
								const SubParser &sub = subParsers.map.value(entryId);
								loadBack(cursor, buffStart, sub.parser->preParserAddOn);
								loadBackC_str(cursor, buffStart, esc); // The remaining part of the escaped sequence is inserted right before the preParserAddOn;
								if (sub.preserveEntryPoint)
									loadBackC_str(cursor, buffStart, tmpStart);
								token.entry = tmpStart;
								sub.parser->parse(token.subTokens, cursor, buffStart, tmpStart, escapedBuffer, tmpEnd, escEnd, token.exit);
								tokenEnded = true;
								break;
							}
							else {
								bool shouldInc = true;
								ByteSet::NodeId exitId = exitPoints.findString(tmpStart);
								if (!BYTE_SET_NOT_A_NODE(exitId)) {
									shouldInc = false;
									const ByteSet::Flags &exitFlags = exitPoints.flags(exitId);
									if (exitFlags & TOKEN_EXIT) {
										// Here we return controll to the upper parser;
										loadBack(cursor, buffStart, postParserAddOn);
									}
									noEscape = cursor;
									loadBackC_str(cursor, buffStart, esc); // The remaining part of the escaped sequence is inserted right before the postParserAddOn;
									if (exitFlags & TOKEN_EXIT) {
										if (exitFlags & PRESERVE)
											loadBackC_str(cursor, buffStart, tmpStart);
										exitPnt = tmpStart;
										addToken(output, token);
										return;
									}
									else if (exitFlags & BREAK_TOKEN) {
										tokenEnded = true;
										break;
									}
								}
								else while (tmpStart != tmp) {
									ByteSet::NodeId entryPref = subParsers.prefixes.findString(tmpStart);
									ByteSet::NodeId exitPref = exitPointPreffixes.findString(tmpStart);
									if (BYTE_SET_NOT_A_NODE(entryPref) && BYTE_SET_NOT_A_NODE(exitPref)) {
										token.text += (*tmpStart);
										tmpStart++;
										escapedBuffer++;
										shouldInc = false;
									}
									else break;
								}
								if ((*esc) == '\0') break;
								if (shouldInc) {
									(*tmp) = (*esc);
									esc++;
									tmp++;
								}
							}
						}
					}
					if (tokenEnded) break;
					if (!escapedChanged) {
						if ((*cursor) == '\0') break;
						(*tmpCursor) = (*cursor);
						cursor++;
						tmpCursor++;
					}
				}
			}
			addToken(output, token);
		}
	}

}
