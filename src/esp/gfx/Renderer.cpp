// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "Renderer.h"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>

#include <Corrade/Containers/StridedArrayView.h>
#include <Magnum/GL/Buffer.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Framebuffer.h>
#include <Magnum/GL/PixelFormat.h>
#include <Magnum/GL/Renderbuffer.h>
#include <Magnum/GL/RenderbufferFormat.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/Image.h>
#include <Magnum/ImageView.h>
#include <Magnum/PixelFormat.h>
#include <Magnum/Platform/GLContext.h>

#include "esp/gfx/DepthUnprojection.h"
#include "esp/gfx/RenderTarget.h"
#include "esp/gfx/magnum.h"
#include "esp/sensor/VisualSensor.h"

namespace Mn = Magnum;

namespace esp {
namespace gfx {

struct BackgroundRenderThread {
  BackgroundRenderThread(WindowlessContext* context)
      : context_{context}, exit_{0}, done_{0} {
    pthread_barrier_init(&startBarrier_, nullptr, 2);
    context_->release();
    t = std::thread(&BackgroundRenderThread::run, this);
    jobsWaiting_ = 1;
    waitThread();
    context_->makeCurrent();
  }

  ~BackgroundRenderThread() {
    exit_.store(1);
    pthread_barrier_wait(&startBarrier_);
    t.join();
    pthread_barrier_destroy(&startBarrier_);
  }

  void waitThread() {
    while (done_.load(std::memory_order_acquire) != jobsWaiting_)
      asm volatile("pause" ::: "memory");

    done_.store(0);
    jobsWaiting_ = 0;
  }

  void submitJob(sensor::VisualSensor& sensor,
                 scene::SceneGraph& sceneGraph,
                 const Mn::MutableImageView2D& view,
                 RenderCamera::Flags flags) {
    jobs_.emplace_back(std::ref(sensor), std::ref(sceneGraph), std::cref(view),
                       flags);
    jobsWaiting_ += 1;
  }

  void startJobs() { pthread_barrier_wait(&startBarrier_); }

  void run() {
    context_->makeCurrentPlatform();
    threadContext_ = new Mn::Platform::GLContext{Mn::NoCreate};
    if (!threadContext_->tryCreate())
      Mn::Fatal{} << "BackgroundRenderThread: Failed to create OpenGL context";

    Mn::GL::Context::makeCurrent(threadContext_);

    Mn::GL::Renderer::enable(Mn::GL::Renderer::Feature::DepthTest);
    Mn::GL::Renderer::enable(Mn::GL::Renderer::Feature::FaceCulling);

    Mn::GL::Context::makeCurrent(nullptr);
    context_->releasePlatform();

    done_.store(1, std::memory_order_release);
    while (true) {
      pthread_barrier_wait(&startBarrier_);

      if (exit_.load() != 0)
        break;

      context_->makeCurrentPlatform();
      Mn::GL::Context::makeCurrent(threadContext_);
      for (auto& job : jobs_) {
        sensor::VisualSensor& sensor = std::get<0>(job);
        scene::SceneGraph& sg = std::get<1>(job);
        RenderCamera::Flags flags = std::get<3>(job);

        if (!(flags & RenderCamera::Flag::ObjectsOnly))
          sensor.renderTarget().renderEnter();

        auto* camera = sensor.getRenderCamera();

        for (auto& it : sg.getDrawableGroups()) {
          // TODO: remove || true
          if (it.second.prepareForDraw(*camera) || true) {
            camera->draw(it.second, flags);
          }
        }

        if (!(flags & RenderCamera::Flag::ObjectsOnly))
          sensor.renderTarget().renderExit();
      }

      for (auto& job : jobs_) {
        sensor::VisualSensor& sensor = std::get<0>(job);
        const Mn::MutableImageView2D& view = std::get<2>(job);
        RenderCamera::Flags flags = std::get<3>(job);
        if (flags & RenderCamera::Flag::ObjectsOnly)
          continue;

        auto sensorType = sensor.specification()->sensorType;
        if (sensorType == sensor::SensorType::Color)
          sensor.renderTarget().readFrameRgba(view);

        if (sensorType == sensor::SensorType::Depth)
          sensor.renderTarget().readFrameDepth(view);

        if (sensorType == sensor::SensorType::Semantic)
          sensor.renderTarget().readFrameObjectId(view);
      }

      Mn::GL::Context::makeCurrent(nullptr);
      context_->releasePlatform();

      int jobsDone = jobs_.size();
      jobs_.clear();
      done_.fetch_add(jobsDone, std::memory_order_release);
    };

    delete threadContext_;
  }

  WindowlessContext* context_;

  std::atomic<int> exit_;
  std::atomic<int> done_;
  std::thread t;
  pthread_barrier_t startBarrier_;

  Mn::Platform::GLContext* threadContext_;
  std::vector<std::tuple<std::reference_wrapper<sensor::VisualSensor>,
                         std::reference_wrapper<scene::SceneGraph>,
                         std::reference_wrapper<const Mn::MutableImageView2D>,
                         RenderCamera::Flags>>
      jobs_;
  int jobsWaiting_ = 0;
};

struct Renderer::Impl {
  explicit Impl(WindowlessContext* context, Flags flags)
      : context_{context}, depthShader_{nullptr}, flags_{flags} {
    Mn::GL::Renderer::enable(Mn::GL::Renderer::Feature::DepthTest);
    Mn::GL::Renderer::enable(Mn::GL::Renderer::Feature::FaceCulling);

    backgroundRenderer_ = new BackgroundRenderThread{context_};
  }

  ~Impl() {
    delete backgroundRenderer_;
    LOG(INFO) << "Deconstructing Renderer";
  }

  void draw(RenderCamera& camera,
            scene::SceneGraph& sceneGraph,
            RenderCamera::Flags flags) {
    for (auto& it : sceneGraph.getDrawableGroups()) {
      // TODO: remove || true
      if (it.second.prepareForDraw(camera) || true) {
        camera.draw(it.second, flags);
      }
    }
  }

  void draw(sensor::VisualSensor& visualSensor,
            scene::SceneGraph& sceneGraph,
            RenderCamera::Flags flags) {
    draw(*visualSensor.getRenderCamera(), sceneGraph, flags);
  }

  void drawAsync(sensor::VisualSensor& visualSensor,
                 scene::SceneGraph& sceneGraph,
                 const Mn::MutableImageView2D& view,
                 RenderCamera::Flags flags) {
    if (contextIsOwned_) {
      context_->release();
      contextIsOwned_ = false;
    }
    backgroundRenderer_->submitJob(visualSensor, sceneGraph, view, flags);
  }

  void startDrawJobs() { backgroundRenderer_->startJobs(); }

  void drawWait() {
    backgroundRenderer_->waitThread();
    if (!contextIsOwned_) {
      context_->makeCurrent();
      contextIsOwned_ = true;
    }
  }

  void bindRenderTarget(sensor::VisualSensor& sensor) {
    auto depthUnprojection = sensor.depthUnprojection();
    if (!depthUnprojection) {
      throw std::runtime_error(
          "Sensor does not have a depthUnprojection matrix");
    }

    if (!depthShader_) {
      depthShader_ = std::make_unique<DepthShader>(
          DepthShader::Flag::UnprojectExistingDepth);
    }

    sensor.bindRenderTarget(RenderTarget::create_unique(
        sensor.framebufferSize(), *depthUnprojection, depthShader_.get(),
        flags_));
  }

 private:
  WindowlessContext* context_;
  bool contextIsOwned_ = true;
  std::unique_ptr<DepthShader> depthShader_;
  const Flags flags_;

  BackgroundRenderThread* backgroundRenderer_ = nullptr;
};

Renderer::Renderer(WindowlessContext* context, Flags flags)
    : pimpl_(spimpl::make_unique_impl<Impl>(context, flags)) {}

void Renderer::draw(RenderCamera& camera,
                    scene::SceneGraph& sceneGraph,
                    RenderCamera::Flags flags) {
  pimpl_->draw(camera, sceneGraph, flags);
}

void Renderer::draw(sensor::VisualSensor& visualSensor,
                    scene::SceneGraph& sceneGraph,
                    RenderCamera::Flags flags) {
  pimpl_->draw(visualSensor, sceneGraph, flags);
}

void Renderer::drawAsync(sensor::VisualSensor& visualSensor,
                         scene::SceneGraph& sceneGraph,
                         const Mn::MutableImageView2D& view,
                         RenderCamera::Flags flags) {
  pimpl_->drawAsync(visualSensor, sceneGraph, view, flags);
}

void Renderer::drawWait() {
  pimpl_->drawWait();
}

void Renderer::startDrawJobs() {
  pimpl_->startDrawJobs();
}

void Renderer::bindRenderTarget(sensor::VisualSensor& sensor) {
  pimpl_->bindRenderTarget(sensor);
}

}  // namespace gfx
}  // namespace esp
