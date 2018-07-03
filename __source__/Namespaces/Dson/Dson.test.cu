#include "Dson.test.h"
#include "Dson.h"
#include "../Tests/Tests.h"

namespace DsonTest {
	namespace {
		void testToStringGauntlet() {
			{
				Dson::Dict dict;
				std::cout << "Simple empty dict:" << std::endl << dict.toString("    ") << std::endl << std::endl;
			}
			{
				Dson::Dict dict;
				{
					Dson::Number num;
					num.value() = 4.5;
					dict.set("value\"", num);
				}
				std::cout << "Simple one elem dict:" << std::endl << dict.toString("    ") << std::endl << std::endl;
			}
			{
				Dson::Dict dict;
				{
					Dson::Number num;
					num.value() = 3.5;
					dict.set("value_0\\", num);
				}
				{
					Dson::Number num;
					num.value() = -5;
					dict.set("value_1/", num);
				}
				std::cout << "Simple two elem dict:" << std::endl << dict.toString("    ") << std::endl << std::endl;
			}
			{
				Dson::Dict mainDict;
				{
					Dson::Dict dict;
					mainDict.set("dict\b", dict);
				}
				std::cout << "Simple emty dict in dict:" << std::endl << mainDict.toString("    ") << std::endl << std::endl;
			}
			{
				Dson::Dict mainDict;
				{
					Dson::Dict dict;
					{
						Dson::Number num;
						num.value() = -8.9;
						dict.set("value\n", num);
					}
					mainDict.set("dict\f", dict);
				}
				std::cout << "Simple dict in dict:" << std::endl << mainDict.toString("    ") << std::endl << std::endl;
			}
			{
				Dson::Dict mainDict;
				{
					Dson::Dict dict;
					{
						Dson::Number num;
						num.value() = -7.9;
						dict.set("value_0\t", num);
					}
					{
						Dson::Number num;
						num.value() = 8.9;
						dict.set("value_1\n", num);
					}
					mainDict.set("dict\r", dict);
				}
				std::cout << "Simple two elem dict in dict:" << std::endl << mainDict.toString("    ") << std::endl << std::endl;
			}
			{
				Dson::Dict mainDict;
				{
					Dson::Dict dict;
					{
						Dson::Number num;
						num.value() = -7.9;
						dict.set("value_0", num);
					}
					{
						Dson::Number num;
						num.value() = 8.9;
						dict.set("value_1", num);
					}
					mainDict.set("dict", dict);
				}
				{
					Dson::Number num;
					num.value() = -7.9;
					mainDict.set("number", num);
				}
				std::cout << "Simple two elem dict in two elem dict:" << std::endl << mainDict.toString("    ") << std::endl << std::endl;
			}
			{
				Dson::Array arr;
				std::cout << "Simple empty array:" << std::endl << arr.toString("    ") << std::endl << std::endl;
			}
			{
				Dson::Array arr;
				{
					Dson::Number num;
					num.value() = -7.9;
					arr.push(num);
				}
				std::cout << "Simple one elem array:" << std::endl << arr.toString("    ") << std::endl << std::endl;
			}
			{
				Dson::Array arr;
				{
					Dson::Number num;
					num.value() = -7.9;
					arr.push(num);
				}
				{
					Dson::Number num;
					num.value() = 5.5;
					arr.push(num);
				}
				std::cout << "Simple two elem array:" << std::endl << arr.toString("    ") << std::endl << std::endl;
			}
			{
				Dson::Array mainArray;
				{
					Dson::Array arr;
					mainArray.push(arr);
				}
				std::cout << "Simple empty array in array:" << std::endl << mainArray.toString("    ") << std::endl << std::endl;
			}
			{
				Dson::Array mainArray;
				{
					Dson::Array arr;
					{
						Dson::Number num;
						num.value() = 12213.0;
						arr.push(num);
					}
					mainArray.push(arr);
				}
				std::cout << "Simple one elem array in array:" << std::endl << mainArray.toString("    ") << std::endl << std::endl;
			}
			{
				Dson::Array mainArray;
				{
					Dson::Array arr;
					{
						Dson::Number num;
						num.value() = 12213.0;
						arr.push(num);
					}
					{
						Dson::Number num;
						num.value() = 231;
						arr.push(num);
					}
					mainArray.push(arr);
				}
				std::cout << "Simple two elem array in array:" << std::endl << mainArray.toString("    ") << std::endl << std::endl;
			}
			{
				Dson::Array mainArray;
				{
					Dson::Array arr;
					{
						Dson::Number num;
						num.value() = 12213.0;
						arr.push(num);
					}
					{
						Dson::Number num;
						num.value() = 231;
						arr.push(num);
					}
					mainArray.push(arr);
				}
				{
					Dson::Number num;
					num.value() = 324;
					mainArray.push(num);
				}
				std::cout << "Simple two elem array in two elem array:" << std::endl << mainArray.toString("    ") << std::endl << std::endl;
			}
			{
				Dson::Dict sceneDict;
				{
					Dson::Dict meshesDict;
					{
						Dson::Dict meshesEntry;
						{
							Dson::String text;
							text.value() = "main_meshes.obj";
							meshesEntry.set("source", text);
						}
						meshesDict.set("main_meshes", meshesEntry);
					}
					{
						Dson::Dict meshesEntry;
						{
							Dson::String text;
							text.value() = "some_other_meshes.obj";
							meshesEntry.set("source", text);
						}
						meshesDict.set("some_other_meshes", meshesEntry);
					}
					sceneDict.set("meshes", meshesDict);
				}
				{
					Dson::Dict materialsDict;
					{
						Dson::Dict glossyMaterialDict;
						{
							{
								Dson::String text;
								text.value() = "dumb_based";
								glossyMaterialDict.set("type", text);
							}
							{
								Dson::String text;
								text.value() = "glossy";
								glossyMaterialDict.set("preset", text);
							}
							{
								Dson::Array fresnelParams;
								Dson::Number number;
								number.value() = 0.9;
								fresnelParams.push(number);
								number.value() = 0.8;
								fresnelParams.push(number);
								number.value() = 0.7;
								fresnelParams.push(number);
								glossyMaterialDict.set("fresnel", fresnelParams);
							}
						}
						materialsDict.set("glossy_material", glossyMaterialDict);
					}
					sceneDict.set("materials", materialsDict);
				}
				{
					Dson::Dict lights;
					{
						Dson::Dict sun;
						{
							Dson::String type;
							type.value() = "simple_soft_directional";
							sun.set("type", type);
						}
						{
							Dson::Array direction;
							Dson::Number number;
							number.value() = 1;
							direction.push(number);
							number.value() = -1;
							direction.push(number);
							number.value() = 1;
							direction.push(number);
							sun.set("direction", direction);
						}
						{
							Dson::Array color;
							Dson::Number number;
							number.value() = 1;
							color.push(number);
							number.value() = 1;
							color.push(number);
							number.value() = 0.9;
							color.push(number);
							sun.set("color", color);
						}
						{
							Dson::Number softness;
							softness.value() = 0.25;
							sun.set("softness", softness);
						}
						lights.set("sun", sun);
					}
					sceneDict.set("lights", lights);
				}
				{
					Dson::Dict objects;
					{
						Dson::Dict item;
						{
							Dson::String mesh;
							mesh.value() = "objects::box";
							item.set("mesh", mesh);
						}
						{
							Dson::Dict transform;
							{
								Dson::Bool recenter;
								recenter.value() = false;
								transform.set("recenter", recenter);
							}
							{
								Dson::Array position;
								Dson::Number number;
								number.value() = 0;
								position.push(number);
								position.push(number);
								position.push(number);
								transform.set("position", position);
							}
							{
								Dson::Array relative_position;
								Dson::Number number;
								number.value() = 0;
								relative_position.push(number);
								relative_position.push(number);
								relative_position.push(number);
								transform.set("relative_position", relative_position);
							}
							{
								Dson::Array rotation;
								Dson::Number number;
								number.value() = 0;
								rotation.push(number);
								rotation.push(number);
								rotation.push(number);
								transform.set("rotation", rotation);
							}
							{
								Dson::Array scale;
								Dson::Number number;
								number.value() = 1;
								scale.push(number);
								scale.push(number);
								scale.push(number);
								transform.set("rotation", scale);
							}
							item.set("transform", transform);
						}
						{
							Dson::String material;
							material.value() = "glossy_material";
							item.set("material", material);
						}
						objects.set("item_0", item);
					}
					{
						Dson::Dict item;
						{
							Dson::String mesh;
							mesh.value() = "objects::box";
							item.set("mesh", mesh);
						}
						{
							Dson::Dict transform;
							{
								Dson::Array position;
								Dson::Number number;
								number.value() = 32;
								position.push(number);
								number.value() = 0;
								position.push(number);
								number.value() = 32;
								position.push(number);
								transform.set("position", position);
							}
							item.set("transform", transform);
						}
						{
							Dson::String material;
							material.value() = "glossy_material";
							item.set("material", material);
						}
						objects.set("item_1", item);
					}
					sceneDict.set("objects", objects);
				}
				{
					Dson::Dict camera;
					{
						Dson::Dict lense;
						{
							Dson::String type;
							type.value() = "simple_stochastic";
							lense.set("type", type);
						}
						{
							Dson::Number angle;
							angle.value() = 60;
							lense.set("angle", angle);
						}
						camera.set("lense", lense);
					}
					{
						Dson::Dict transform;
						{
							Dson::Array target;
							Dson::Number number;
							number.value() = 0;
							target.push(number);
							target.push(number);
							target.push(number);
							transform.set("target", target);
						}
						{
							Dson::Array rotation;
							Dson::Number number;
							number.value() = 0;
							rotation.push(number);
							rotation.push(number);
							rotation.push(number);
							transform.set("position", rotation);
						}
						{
							Dson::Number distance;
							distance.value() = 128;
							transform.set("distance", distance);
						}
						camera.set("transform", transform);
					}
					sceneDict.set("camera", camera);
				}
				std::cout << "Example scene:" << std::endl << sceneDict.toString("    ") << std::endl << std::endl;
			}
		}
	}


	void testToString() {
		Tests::runTest(testToStringGauntlet, "Testing Dson::toString()");
	}



	namespace {
		static void printIfNotNullAndDispose(Dson::Object *object) {
			if (object != NULL) {
				std::cout << object->toString("  ") << std::endl;
				delete object;
			}
		}
		static void testParseDson(const std::string &source) {
			std::cout << "__________________________________________________" << std::endl;
			std::cout << "Source: " << source << std::endl;
			std::cout << "__________________" << std::endl;
			printIfNotNullAndDispose(Dson::parse(source, &std::cout));
		}

		static void testFromStringGauntlet() {
			testParseDson("");
			testParseDson("{}");
			testParseDson("[]");
			testParseDson("{[]}");
			testParseDson("[{}]");
			testParseDson("[ {}, {}]");
			testParseDson("[ {  } , {  } ]");
			testParseDson("[{},{}]");
			testParseDson("[\"aaa\"]");
			testParseDson("[\"aaa\",]");
			testParseDson("[\"aaa\",\"bbb\"]");
			testParseDson("{\"aaa\"}");
			testParseDson("{\"aaa\":}");
			testParseDson("{\"aaa\":\"bbb\"}");
			testParseDson("{\"aaa\":\"bbb\", \"ccc\"}");
			testParseDson("{\"aaa\":\"bbb\", \"ccc\" :}");
			testParseDson("{\"aaa\":\"bbb\", \"ccc\" : \"ddd\"}");
			testParseDson("{\"aaa\":\"bbb\", \"ccc\" : \"ddd\",  \"aaa\":[]}");
			testParseDson("1");
			testParseDson("-1");
			testParseDson("1.21");
			testParseDson("1junk");
			testParseDson("some_garbage");
			testParseDson("{\"a\":true, \"b\": false, \"c\" : null, \"d\": 0, \"e\": [], \"f\": {}, \"g\": \"text\", \"h\": {\"1\":1}, \"i\":[0], \"j\" : {\"1\":\"1\", \"2\":[{},{}]}}");
		}
	}
	void testFromString() {
		Tests::runTest(testFromStringGauntlet, "Testing Dson::parse()");
	}
}
