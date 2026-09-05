#pragma once

#include <string>

// 简单的 GLSL 着色器程序封装: 从文件或源码编译链接, 提供 use/uniform 查询。
// 所有失败信息通过 logOut 返回, 便于上层写入日志排查渲染问题。
class GLShaderProgram
{
public:
	GLShaderProgram() = default;
	~GLShaderProgram();

	// 从 .vert/.frag 文件加载 (读取失败或编译链接失败返回 false, 详情写入 logOut)
	bool loadFromFiles(const std::string &vertPath, const std::string &fragPath, std::string &logOut);
	// 从源码字符串加载
	bool loadFromSource(const std::string &vertSrc, const std::string &fragSrc, std::string &logOut);

	void use() const;
	unsigned int id() const { return m_prog; }
	bool valid() const { return m_prog != 0; }
	int  uniform(const char *name) const;   // glGetUniformLocation

private:
	unsigned int m_prog = 0;
	// 编译单个 shader; 成功返回 true 并输出 id, 失败把编译日志写入 logOut
	bool compile(unsigned int type, const std::string &src, unsigned int &outShader, std::string &logOut);
	static bool readFile(const std::string &path, std::string &out);
};
