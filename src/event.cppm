export module engine.event;

import std;

namespace engine {

// ---------------------------------------------------------------------------: Categories

export enum class EventCategory : int {
  kNone     = 0,
  kWindow   = 1 << 0,
  kInput    = 1 << 1,
  kKeyboard = 1 << 2,
  kMouse    = 1 << 3,
};

export constexpr int operator|(EventCategory a, EventCategory b) {
  return static_cast<int>(a) | static_cast<int>(b);
}

export constexpr int operator|(int a, EventCategory b) {
  return a | static_cast<int>(b);
}

// ---------------------------------------------------------------------------: Event base

// Polymorphic event interface. Concrete events inherit from EventBase<Self>
// (CRTP) to get GetType() and Clone() for free.
export class Event {
 public:
  virtual ~Event() = default;

  virtual std::type_index GetType() const = 0;
  virtual std::unique_ptr<Event> Clone() const = 0;
  virtual int GetCategoryFlags() const { return 0; }

  bool IsInCategory(EventCategory c) const {
    return (GetCategoryFlags() & static_cast<int>(c)) != 0;
  }

  bool IsHandled() const { return handled_; }
  void SetHandled(bool handled = true) { handled_ = handled; }

 private:
  bool handled_ = false;
};

export template <typename Derived>
class EventBase : public Event {
 public:
  std::type_index GetType() const override { return typeid(Derived); }

  std::unique_ptr<Event> Clone() const override {
    return std::make_unique<Derived>(static_cast<const Derived&>(*this));
  }
};

// ---------------------------------------------------------------------------: Window events

export class WindowResizeEvent : public EventBase<WindowResizeEvent> {
 public:
  WindowResizeEvent(std::uint32_t width, std::uint32_t height) : width_(width), height_(height) {}

  std::uint32_t GetWidth() const { return width_; }
  std::uint32_t GetHeight() const { return height_; }
  int GetCategoryFlags() const override { return static_cast<int>(EventCategory::kWindow); }

 private:
  std::uint32_t width_;
  std::uint32_t height_;
};

export class WindowCloseEvent : public EventBase<WindowCloseEvent> {
 public:
  int GetCategoryFlags() const override { return static_cast<int>(EventCategory::kWindow); }
};

// ---------------------------------------------------------------------------: Keyboard events

export class KeyPressEvent : public EventBase<KeyPressEvent> {
 public:
  KeyPressEvent(int key_code, int mods, bool is_repeat) : key_code_(key_code), mods_(mods), is_repeat_(is_repeat) {}

  int GetKeyCode() const { return key_code_; }
  int GetMods() const { return mods_; }
  bool IsRepeat() const { return is_repeat_; }

  int GetCategoryFlags() const override {
    return EventCategory::kInput | EventCategory::kKeyboard;
  }

 private:
  int key_code_;
  int mods_;
  bool is_repeat_;
};

export class KeyReleaseEvent : public EventBase<KeyReleaseEvent> {
 public:
  KeyReleaseEvent(int key_code, int mods) : key_code_(key_code), mods_(mods) {}

  int GetKeyCode() const { return key_code_; }
  int GetMods() const { return mods_; }

  int GetCategoryFlags() const override {
    return EventCategory::kInput | EventCategory::kKeyboard;
  }

 private:
  int key_code_;
  int mods_;
};

// ---------------------------------------------------------------------------: Mouse events

export class MouseMoveEvent : public EventBase<MouseMoveEvent> {
 public:
  MouseMoveEvent(double x, double y) : x_(x), y_(y) {}

  double GetX() const { return x_; }
  double GetY() const { return y_; }

  int GetCategoryFlags() const override {
    return EventCategory::kInput | EventCategory::kMouse;
  }

 private:
  double x_;
  double y_;
};

export class MouseScrollEvent : public EventBase<MouseScrollEvent> {
 public:
  MouseScrollEvent(double x_offset, double y_offset) : x_offset_(x_offset), y_offset_(y_offset) {}

  double GetXOffset() const { return x_offset_; }
  double GetYOffset() const { return y_offset_; }

  int GetCategoryFlags() const override {
    return EventCategory::kInput | EventCategory::kMouse;
  }

 private:
  double x_offset_;
  double y_offset_;
};

export class MouseButtonPressEvent : public EventBase<MouseButtonPressEvent> {
 public:
  MouseButtonPressEvent(int button, int mods) : button_(button), mods_(mods) {}

  int GetButton() const { return button_; }
  int GetMods() const { return mods_; }

  int GetCategoryFlags() const override {
    return EventCategory::kInput | EventCategory::kMouse;
  }

 private:
  int button_;
  int mods_;
};

export class MouseButtonReleaseEvent : public EventBase<MouseButtonReleaseEvent> {
 public:
  MouseButtonReleaseEvent(int button, int mods) : button_(button), mods_(mods) {}

  int GetButton() const { return button_; }
  int GetMods() const { return mods_; }

  int GetCategoryFlags() const override {
    return EventCategory::kInput | EventCategory::kMouse;
  }

 private:
  int button_;
  int mods_;
};

// ---------------------------------------------------------------------------: Listener

export class EventListener {
 public:
  virtual ~EventListener() = default;
  virtual void OnEvent(Event& event) = 0;
};

// ---------------------------------------------------------------------------: Dispatcher

// Type-safe dispatch: call handler only when the event's runtime type matches T.
// Handler signature: bool(T&) — return true to mark event handled.
export class EventDispatcher {
 public:
  explicit EventDispatcher(Event& event) : event_(event) {}

  template <typename T, typename F>
  bool Dispatch(F&& handler) {
    if (event_.GetType() == typeid(T)) {
      if (std::forward<F>(handler)(static_cast<T&>(event_))) {
        event_.SetHandled(true);
      }
      return true;
    }
    return false;
  }

 private:
  Event& event_;
};

// ---------------------------------------------------------------------------: Bus

// Minimal event bus with immediate dispatch. Listeners are raw pointers, so
// callers are responsible for unregistering before destruction.
export class EventBus {
 public:
  void AddListener(EventListener* listener) {
    if (listener != nullptr) {
      listeners_.push_back(listener);
    }
  }

  void RemoveListener(EventListener* listener) {
    auto it = std::ranges::find(listeners_, listener);
    if (it != listeners_.end()) {
      listeners_.erase(it);
    }
  }

  void Publish(Event& event) {
    for (EventListener* l : listeners_) {
      if (event.IsHandled()) {
        break;
      }
      l->OnEvent(event);
    }
  }

 private:
  std::vector<EventListener*> listeners_;
};

}  // namespace engine
