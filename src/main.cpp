import std;
import app;

int main() {
  try {
    app::Application engine;
    engine.Run();
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
