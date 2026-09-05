#include "stdafx.h"
#include "GLShader.h"
#include <fstream>
#include <sstream>

GLShaderProgram::~GLShaderProgram()
{
	if (m_prog) glDeleteProgram(m_prog);
}

bool GLShaderProgram::readFile(const std::string &path, std::string &out)
{
	std::ifstream fs(path, std::ios::binary);
	if (!fs.is_open()) return false;
	std::ostringstream ss;
	ss << fs.rdbuf();
	out = ss.str();
	return !out.empty();
}

bool GLShaderProgram::compile(unsigned int type, const std::string &src, unsigned int &outShader, std::string &logOut)
{
	GLuint sh = glCreateShader(type);
	const char *p = src.c_str();
	glShaderSource(sh, 1, &p, nullptr);
	glCompileShader(sh);
	GLint ok = 0;
	glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
	if (!ok)
	{
		char buf[2048] = { 0 };
		GLsizei n = 0;
		glGetShaderInfoLog(sh, sizeof(buf) - 1, &n, buf);
		logOut += std::string(type == GL_VERTEX_SHADER ? "[顶点着色器编译失败] " : "[片元着色器编译失败] ") + buf + "\n";
		glDeleteShader(sh);
		return false;
	}
	outShader = sh;
	return true;
}

bool GLShaderProgram::loadFromSource(const std::string &vertSrc, const std::string &fragSrc, std::string &logOut)
{
	unsigned int vs = 0, fs = 0;
	if (!compile(GL_VERTEX_SHADER, vertSrc, vs, logOut)) return false;
	if (!compile(GL_FRAGMENT_SHADER, fragSrc, fs, logOut)) { glDeleteShader(vs); return false; }

	GLuint prog = glCreateProgram();
	glAttachShader(prog, vs);
	glAttachShader(prog, fs);
	glLinkProgram(prog);
	glDeleteShader(vs);
	glDeleteShader(fs);

	GLint ok = 0;
	glGetProgramiv(prog, GL_LINK_STATUS, &ok);
	if (!ok)
	{
		char buf[2048] = { 0 };
		GLsizei n = 0;
		glGetProgramInfoLog(prog, sizeof(buf) - 1, &n, buf);
		logOut += std::string("[着色器程序链接失败] ") + buf + "\n";
		glDeleteProgram(prog);
		return false;
	}
	if (m_prog) glDeleteProgram(m_prog);
	m_prog = prog;
	logOut += "着色器程序链接成功\n";
	return true;
}

bool GLShaderProgram::loadFromFiles(const std::string &vertPath, const std::string &fragPath, std::string &logOut)
{
	std::string vsrc, fsrc;
	if (!readFile(vertPath, vsrc)) { logOut += "无法读取顶点着色器文件: " + vertPath + "\n"; return false; }
	if (!readFile(fragPath, fsrc)) { logOut += "无法读取片元着色器文件: " + fragPath + "\n"; return false; }
	logOut += "已读取着色器文件: " + vertPath + " , " + fragPath + "\n";
	return loadFromSource(vsrc, fsrc, logOut);
}

void GLShaderProgram::use() const
{
	if (m_prog) glUseProgram(m_prog);
}

int GLShaderProgram::uniform(const char *name) const
{
	return m_prog ? glGetUniformLocation(m_prog, name) : -1;
}
