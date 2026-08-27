//=============================================================================
//
//  The contents of this file are covered by the Viskores license. See
//  LICENSE.txt for details.
//
//  By contributing to this file, all contributors agree to the Developer
//  Certificate of Origin Version 1.1 (DCO 1.1) as stated in DCO.txt.
//
//=============================================================================

#include "TransferFunction1D.h"
#include "array/ArrayConversion.h"
// Viskores
#include <viskores/cont/ArrayExtractComponent.h>
#include <viskores/cont/ArrayHandleConstant.h>
#include <viskores/cont/ArrayHandleStride.h>
#include <viskores/rendering/CanvasRayTracer.h>
#include <viskores/rendering/raytracing/Camera.h>
#include <viskores/rendering/raytracing/Ray.h>
#include <viskores/rendering/raytracing/RayOperations.h>
#include <viskores/rendering/raytracing/VolumeRendererStructured.h>

namespace
{

template <typename ComponentType>
void FillColorTable(viskores::cont::ColorTable& table,
                    const viskores::cont::UnknownArrayHandle& array)
{
  viskores::Id numValues = array.GetNumberOfValues();

  std::array<viskores::cont::ArrayHandleStride<ComponentType>, 3> colorChannels;
  colorChannels[0] = array.ExtractComponent<ComponentType>(0);
  if (array.GetNumberOfComponentsFlat() > 1)
  {
    colorChannels[1] = array.ExtractComponent<ComponentType>(1);
  }
  else
  {
    colorChannels[1] = viskores::cont::ArrayExtractComponent(
      viskores::cont::ArrayHandleConstant<ComponentType>(0, numValues), 0);
  }
  if (array.GetNumberOfComponentsFlat() > 2)
  {
    colorChannels[2] = array.ExtractComponent<ComponentType>(2);
  }
  else
  {
    colorChannels[2] = viskores::cont::ArrayExtractComponent(
      viskores::cont::ArrayHandleConstant<ComponentType>(0, numValues), 0);
  }

  bool hasOpacity;
  viskores::cont::ArrayHandleStride<ComponentType> alphaChannel;
  if (array.GetNumberOfComponentsFlat() > 3)
  {
    alphaChannel = array.ExtractComponent<ComponentType>(3);
    hasOpacity = true;
  }
  else
  {
    hasOpacity = false;
  }

  std::array<typename viskores::cont::ArrayHandleStride<ComponentType>::ReadPortalType, 3>
    colorPortals;
  std::transform(colorChannels.begin(),
                 colorChannels.end(),
                 colorPortals.begin(),
                 [](auto array) { return array.ReadPortal(); });
  typename viskores::cont::ArrayHandleStride<ComponentType>::ReadPortalType alphaPortal;
  if (hasOpacity)
  {
    alphaPortal = alphaChannel.ReadPortal();
  }
  if (numValues > 1)
  {
    viskores::Float64 scale = 1.0 / (numValues - 1);
    for (viskores::Id index = 0; index < numValues; ++index)
    {
      std::array<viskores::Float32, 3> color;
      std::transform(colorPortals.begin(),
                     colorPortals.end(),
                     color.begin(),
                     [index](auto portal)
                     { return static_cast<viskores::Float32>(portal.Get(index)); });
      table.AddPoint(index * scale, { color[0], color[1], color[2] });
      if (hasOpacity)
      {
        table.AddPointAlpha(index * scale, static_cast<viskores::Float32>(alphaPortal.Get(index)));
      }
    }
  }
  else
  {
    // Special case: only one color given in array.
    std::array<viskores::Float32, 3> color;
    std::transform(colorPortals.begin(),
                   colorPortals.end(),
                   color.begin(),
                   [](auto portal) { return static_cast<viskores::Float32>(portal.Get(0)); });
    table.AddPoint(0, { color[0], color[1], color[2] });
    table.AddPoint(1, { color[0], color[1], color[2] });
    if (hasOpacity)
    {
      viskores::Float32 alpha = static_cast<viskores::Float32>(alphaPortal.Get(0));
      table.AddPointAlpha(0, alpha);
      table.AddPointAlpha(1, alpha);
    }
  }
}

} // namespace

namespace viskores_device
{

TransferFunction1D::TransferFunction1D(ViskoresDeviceGlobalState* d)
  : Volume(d)
  , m_spatialField(this)
  , m_colorArray(this)
  , m_opacityArray(this)
{
}

void TransferFunction1D::commitParameters()
{
  this->Volume::commitParameters();

  this->m_spatialField = getParamObject<SpatialField>("value");

  this->m_unitDistance = getParam("unitDistance", 1.0f);

  this->m_colorArray = this->getParamObject<Array1D>("color");
  this->m_color = { 1, 1, 1, 1 };
  this->getParam("color", ANARI_FLOAT32_VEC4, &this->m_color);
  this->getParam("color", ANARI_FLOAT32_VEC3, &this->m_color);

  this->m_opacityArray = this->getParamObject<Array1D>("opacity");
  this->m_alpha = 1.0f;
  this->getParam("opacity", ANARI_FLOAT32, &this->m_alpha);

  box1 range = { 0, 1 };
  this->getParam("valueRange", ANARI_FLOAT32_BOX1, &range);
  this->m_valueRange = { range.lower, range.upper };
}

void TransferFunction1D::finalize()
{
  Volume::finalize();

  if (!this->m_spatialField)
  {
    reportMessage(ANARI_SEVERITY_WARNING, "'transferFunction1D' volume missing 'value' parameter");
    return;
  }

  // Determine sample distance (which also affects colors).
  viskores::cont::DataSet dataSet = this->m_spatialField->getDataSet();
  viskores::Bounds bounds = dataSet.GetCoordinateSystem().GetBounds();
  viskores::Float32 diagonalLength =
    static_cast<viskores::Float32>(viskores::Magnitude(bounds.MaxCorner() - bounds.MinCorner()));
  constexpr viskores::IdComponent numberOfSamples = 200;
  this->m_sampleDistance = static_cast<viskores::Float32>(diagonalLength / numberOfSamples);
  if (this->m_unitDistance <= 0.f)
  {
    this->reportMessage(ANARI_SEVERITY_WARNING,
                        "'unitDistance' must be positive; using a small positive value");
    this->m_unitDistance = 1e-6f;
  }

  // Reset and fill color table
  viskores::cont::ColorTable colorTable{ viskores::ColorSpace::RGB };
  bool colorsHaveAlpha = false;
  if (this->m_colorArray)
  {
    // Convert to Viskores colors
    viskores::cont::UnknownArrayHandle viskoresColors =
      ANARIColorsToViskoresColors(this->m_colorArray->dataAsViskoresArray());
    if (viskoresColors.GetNumberOfComponentsFlat() > 3)
    {
      colorsHaveAlpha = true;
    }

    // Copy colors into ColorTable
    // NOTE: I am not at all convinced that this is a good idea. If we are
    // getting an array, that probably means that the client has already sampled
    // the colors to the desired level, and we should just use that array.
    // Instead, we are building piecewise linear segments and then resampling
    // again.
    if (viskoresColors.IsBaseComponentType<viskores::Float32>())
    {
      FillColorTable<viskores::Float32>(colorTable, viskoresColors);
    }
    else if (viskoresColors.IsBaseComponentType<viskores::Float64>())
    {
      FillColorTable<viskores::Float64>(colorTable, viskoresColors);
    }
    else
    {
      reportMessage(ANARI_SEVERITY_ERROR, "Unexpected type for color table data.");
    }
  }
  else
  {
    colorTable.AddPoint(0, { m_color[0], m_color[1], m_color[2] });
    colorTable.AddPoint(1, { m_color[0], m_color[1], m_color[2] });
  }

  if (m_opacityArray)
  {
    if (colorsHaveAlpha)
    {
      reportMessage(ANARI_SEVERITY_WARNING, "Alpha given in both color and opacity parameters.");
    }
    if (m_opacityArray->size() > 1)
    {
      viskores::Float64 scale = 1.0 / (m_opacityArray->size() - 1);
      for (size_t index = 0; index < m_opacityArray->size(); ++index)
      {
        float opacity = *m_opacityArray->valueAt<float>(index);
        colorTable.AddPointAlpha(index * scale, opacity);
      }
    }
    else
    {
      float opacity = *m_opacityArray->valueAt<float>(1);
      colorTable.AddPointAlpha(0, opacity);
      colorTable.AddPointAlpha(1, opacity);
    }
  }
  else if (!colorsHaveAlpha)
  {
    colorTable.AddPointAlpha(0, this->m_alpha);
    colorTable.AddPointAlpha(1, this->m_alpha);
  }

  colorTable.RescaleToRange(this->m_valueRange);

  // Now that we have the color table, build a simple map array that the render caster
  // can use to convert fields to colors.
  constexpr viskores::Float32 conversionToFloatSpace = (1.0f / 255.0f);

  viskores::cont::ArrayHandle<viskores::Vec4ui_8> temp;

  {
    viskores::cont::ScopedRuntimeDeviceTracker tracker(viskores::cont::DeviceAdapterTagSerial{});
    colorTable.Sample(1024, temp);
  }

  this->m_colorMap.Allocate(1024);
  auto portal = this->m_colorMap.WritePortal();
  auto colorPortal = temp.ReadPortal();
  const viskores::Float32 alphaSampleDistance = this->m_sampleDistance / this->m_unitDistance;
  for (viskores::Id i = 0; i < 1024; ++i)
  {
    auto color = colorPortal.Get(i);
    viskores::Float32 opacity = color[3] * conversionToFloatSpace;
    // The opacity given is based on the unit distance. We need to scale that
    // to be the opacity for the sampling distance of the ray caster. This
    // scaling is nonlinear.
    //
    // Opacity (α) scales exponentially with respect to the distance the ray
    // travels through the material (d).
    //
    // α = 1 - exp(-τ·d)
    //
    // where τ is the "transparency coefficient" based on the absorption of the
    // material, which is independent of the distance. Flipping this relationship,
    // we get
    //
    // τ = -(1/d)·ln(1-α).
    //
    // We are given a unit distance, d_u, and an opacity based on that
    // sampling distance, α_u. That means τ = -(1/d_u)·ln(1-α_u). Returning to
    // the first equation defining α and substituting this τ and the ray caster's
    // sample distance, d_s, we get the following opacity for the sample
    // distance.
    //
    // α_s = 1 - (1 - α_u)^(d_s/d_u)
    opacity = opacity >= 1.f ? 1.f : 1.f - viskores::Pow(1.f - opacity, alphaSampleDistance);
    viskores::Vec4f_32 t(color[0] * conversionToFloatSpace,
                         color[1] * conversionToFloatSpace,
                         color[2] * conversionToFloatSpace,
                         opacity);
    portal.Set(i, t);
  }
}

void TransferFunction1D::render(viskores::rendering::Canvas& canvas,
                                const viskores::rendering::Camera& camera) const
{
  viskores::cont::DataSet dataSet = this->m_spatialField->getDataSet();
  const viskores::cont::Field& field = dataSet.GetField("data");
  viskores::cont::CoordinateSystem coords = dataSet.GetCoordinateSystem();

  viskores::cont::CellSetStructured<3> cellSet;
  try
  {
    dataSet.GetCellSet().AsCellSet(cellSet);
  }
  catch (viskores::cont::ErrorBadType)
  {
    this->reportMessage(ANARI_SEVERITY_ERROR,
                        "Transfer function 1D volume has bad cell set type from spatial field");
    return;
  }

  viskores::rendering::CanvasRayTracer* canvasRT =
    dynamic_cast<viskores::rendering::CanvasRayTracer*>(&canvas);
  if (canvasRT == nullptr)
  {
    this->reportMessage(ANARI_SEVERITY_ERROR, "Bad canvas detected for TransferFunction1D.");
    return;
  }

  viskores::rendering::raytracing::VolumeRendererStructured tracer;

  viskores::Int32 width = (viskores::Int32)canvas.GetWidth();
  viskores::Int32 height = (viskores::Int32)canvas.GetHeight();
  viskores::rendering::raytracing::Camera rayCamera = camera.CreateRaytracingCamera(width, height);

  viskores::rendering::raytracing::Ray<viskores::Float32> rays;
  rayCamera.CreateRays(rays, coords.GetBounds());
  rays.Buffers.at(0).InitConst(0.f);
  viskores::rendering::raytracing::RayOperations::MapCanvasToRays(
    rays, camera.CreateRaytracingCamera(width, height), canvasRT->GetDepthBuffer());

  tracer.SetSampleDistance(this->m_sampleDistance);

  tracer.SetData(coords, field, cellSet, this->m_valueRange);
  tracer.SetColorMap(this->m_colorMap);

  tracer.Render(rays);

  canvasRT->WriteToCanvas(rays, rays.Buffers.at(0).Buffer, camera, false);
}

const SpatialField* TransferFunction1D::spatialField() const
{
  return this->m_spatialField.get();
}

viskores::Bounds TransferFunction1D::bounds() const
{
  return isValid() ? this->m_spatialField->getDataSet().GetCoordinateSystem().GetBounds()
                   : viskores::Bounds();
}

bool TransferFunction1D::isValid() const
{
  return this->m_spatialField;
}

} // namespace viskores_device
