#pragma once
#include <string>
#include "ByteMap.h"


namespace Parser {
	
	class RecursiveParser {
	public:
		typedef std::string Keyword;
		typedef std::string Text;

		struct Token {
			Text text;
			Keyword entry;
			std::vector<Token> subTokens;
			Keyword exit;
			void dump(std::ostream &stream)const;
			static void dump(const std::vector<Token> &tokens, std::ostream &stream);
		};

		enum EscapeType {
			ESCAPE_REPLACE,
			ESCAPE_DIRECT_TO_TEXT,
			ESCAPE_RECURSIVE
		};



	public:
		RecursiveParser();
		~RecursiveParser();

		/*
		Adds an escape sequence;
		Params:
			sequence - pattern, that should be replaces;
			replacement - this one will be inserted when the sequence is encountered;
			type - escape sequences may be treated in several different ways:
				1. ESCAPE_REPLACE - will cause the parser to simply replace the sequence
				and re-analize it without any second escape checking;
				2. ESCAPE_DIRECT_TO_TEXT - will cause the parser to transfer the sequence
				directly to the text field of the token, therefore, escaping any potential keyword detection;
				(might be useful for things like "\"");
				3. ESCAPE_RECURSIVE - will replace the sequence and force the parser
				to re-analize for possible second escapes.
				(for example, addEscapeSequence("  ", " ", ESCAPE_RECURSIVE) will turn
				any number of spaces into one).
		Notes:
			1. If replacements are longer than sequences and the logic is recursive,
			one or more escape sequences can get intertwined and cause the program
			to crash. Since there's no protection from this built in,
			please, use this one with caution;
			2. Adding the same sequence twice will just overwrite the previous result.
		*/
		void addEscapeSequence(const Text &sequence, const Text &replacement, EscapeType type);

		/*
		Adds an ignored sequence;
		Param:
			sequence - the pattern, likes of which should be removed.
		Note:
			This one is pretty much the same as addEscapeSequence(sequence, "", ESCAPE_REPLACE);
			So, for further description, read the comment above addEscapeSequence().
		*/
		void addIgnoredSequence(const Text &sequence);

		/*
		Adds an entry point;
		Params:
			key - the entry point.
			preserve - if true, the entry point will be re-parsed when the parser "takes over".
		Note:
			When this key is encountered in the other parser containing this one,
			the parsing process control will be transfered to this one to fill the subTokens of the token that's being parsed.
		*/
		void addEntryPoint(const Keyword &key, bool preseve);

		/*
		Adds an exit point;
		Params:
			key - the exit point.
			preserve - if true, the exit point will be re-parsed when the control is returned to the upper parser.
		Note:
			When the given pattern is encountered, the parser stops parsing and returns control to the caller;
			Therefore, if this is the top parser, the parsing process might not go throught all the way.
		*/
		void addExitPoint(const Keyword &key, bool preserve);

		/*
		Adds a delimiter;
		Param:
			key - whenever this is encountered, the parser will move on to the next token.
				(It's effectively the same as a delimiter of some kind)
		*/
		void addDelimiter(const Keyword &key);

		/*
		Sets a pre-parse add on text;
		Param:
			text - this will be added before the text.
		Note:
			"Pre-parse add on" virtually will sit between the entry point and the content that should be parsed;
			No matter what, this text will be parsed.
		*/
		void setPreParseAddOn(const Text &text);

		/*
		Sets a post-parse add on text;
		Param:
			text - this will be added after the exit point.
		Note:
			"Post-parse add on" virtually will sit after the exit point of the segment that was parsed;
			No matter what, this text will be parsed by the upper parser(if there is any).
		*/
		void setPostParseAddOn(const Text &text);

		/*
		Adds a sub-parser;
		Param:
			parser - the sub-parser.
		Note:
			Since there is no way to check if the parsers passed here still exist,
			it is extremely unsafe to use the parser if any single node in the parser network is or might be deallocated.
		*/
		void addSubParser(const RecursiveParser *parser);

		/*
		Parses the input string;
		Param:
			input - the string that should be parsed.
		*/
		std::vector<Token> parse(const Text &input)const;
		/*
		Parses the input string;
		Params:
			input - the string that should be parsed;
			output - parser output (tokens will merely be appended to the output).
		*/
		void parse(const Text &input, std::vector<Token> &output)const;



	private:
		template<typename Type>
		struct KeywordMap {
			ByteMap<Type> map;
			ByteSet prefixes;
		};
		struct EscapeValue {
			Text replacement;
			EscapeType type;
		};
		struct EntryPoint {
			Keyword key;
			bool preserve;
		};
		struct SubParser {
			const RecursiveParser* parser;
			bool preserveEntryPoint;
		};

		KeywordMap<EscapeValue> escapeSequences;
		std::vector<EntryPoint> entryPoints;
		ByteSet exitPoints;
		ByteSet exitPointPreffixes;
		Text preParserAddOn;
		Text postParserAddOn;
		size_t maxEscapeScale;
		size_t maxPreParserAddOnSize;
		size_t maxPostParserAddOnSize;
		KeywordMap<SubParser> subParsers;

		void addExitPointOrTokenBreaker(const Keyword &key, const ByteSet::Flags &flags);
		void parse(std::vector<Token> &output, char *&cursor, char *buffStart, char *&tmpBuffer, char *&escapedBuffer, const char *tmpEnd, const char *escEnd, std::string &exitPnt)const;
	};


}
