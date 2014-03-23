#ifndef UTILS_H
#define UTILS_H

#include <map>
#include <string>
#include <fstream>
#include <iostream>
#include <exception>

std::map<std::string, int> loadDB(const std::string &dbfilename)
{
	std::map<std::string, int> DB;

	std::ifstream dbfile(dbfilename);

	if(dbfile.is_open())
	{
		std::string line;
		dbfile >> line;

		while(!dbfile.eof())
		{
			dbfile >> line;
			std::istringstream ss(line);

			std::string name;
			std::getline(ss, name, ',');

			int cls;
			ss >> cls;

			DB[name] = cls;
		}
		dbfile.close();
	}

	return DB;
}

bool checkClass(const std::string &img, int cls)
{
	bool in = false;
	try
	{
		static std::map<std::string, int> DB = loadDB("../data/classes.csv");

		in = (DB.at(img) == cls);
	}
	catch(std::out_of_range &e)
	{
		std::cout << "Out of range excpetion with " << img << std::endl;
	}
	catch(std::exception &e)
	{
		std::cout << e.what() << std::endl;
	}

	return in;
}

#endif