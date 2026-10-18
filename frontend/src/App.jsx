import React, { useCallback, useEffect, useRef, useState } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "./App.css";

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"
).replace(/\/+$/, "");

function formatPersistence(days) {
  const totalMinutes = Math.max(0, Math.round(Number(days || 0) * 24 * 60));
  const hours = Math.floor(totalMinutes / 60);
  const minutes = totalMinutes % 60;

  if (hours === 0) return `${minutes}m`;
  if (minutes === 0) return `${hours}h`;
  return `${hours}h ${minutes}m`;
}

function formatDistance(distance) {
  const value = Number(distance);
  if (!Number.isFinite(value)) return "N/A";
  if (value < 1) return `${Math.round(value * 1000)} m`;
  return `${value.toFixed(1)} km`;
}

function formatEventType(predictedClass) {
  const labels = {
    industrial_fire: "Industrial Fire",
    natural_fire: "Natural Fire",
    agricultural_burning: "Agricultural Burning",
    persistent_thermal_source: "Persistent Thermal Source",
    gas_flare: "Gas Flare",
    uncertain: "Uncertain Thermal Event",
  };

  return labels[predictedClass] || "Thermal Anomaly";
}

function formatEventLocation(event) {
  const locality = [event.city, event.state].filter(Boolean).join(", ");
  if (locality) return locality;

  if (Number.isFinite(event.lat) && Number.isFinite(event.lng)) {
    return `${event.lat.toFixed(2)}°, ${event.lng.toFixed(2)}°`;
  }

  return event.location || "Location unavailable";
}

function formatTimeAgo(timestamp) {
  if (!timestamp) return "not recorded";

  const elapsedSeconds = Math.max(
    0,
    Math.floor((Date.now() - new Date(timestamp).getTime()) / 1000)
  );
  if (elapsedSeconds < 60) return "just now";

  const elapsedMinutes = Math.floor(elapsedSeconds / 60);
  if (elapsedMinutes < 60) return `${elapsedMinutes}m ago`;

  const elapsedHours = Math.floor(elapsedMinutes / 60);
  if (elapsedHours < 24) return `${elapsedHours}h ago`;

  return `${Math.floor(elapsedHours / 24)}d ago`;
}

function getFeedIndicator(status, statusUnavailable) {
  if (statusUnavailable) {
    return { label: "STATUS OFFLINE", tone: "error" };
  }
  if (!status) {
    return { label: "CHECKING FEED", tone: "unknown" };
  }
  if (status.status === "running") {
    return { label: "SYNCING", tone: "running" };
  }
  if (status.status === "failed") {
    return { label: "SYNC FAILED", tone: "error" };
  }
  if (status.status === "partial") {
    return { label: "PARTIAL UPDATE", tone: "warning" };
  }
  if (status.status === "never_run") {
    return { label: "AWAITING SYNC", tone: "unknown" };
  }
  if (status.status !== "success" || !status.last_success_at) {
    return { label: "FEED UNKNOWN", tone: "unknown" };
  }

  const ageMs = Date.now() - new Date(status.last_success_at).getTime();
  const staleAfterMs = (status.interval_seconds || 15 * 60) * 2 * 1000;
  if (!Number.isFinite(ageMs) || ageMs > staleAfterMs) {
    return { label: "DATA DELAYED", tone: "warning" };
  }

  return {
    label: `UPDATED ${formatTimeAgo(status.last_success_at)}`,
    tone: "healthy",
  };
}

function intensityFromFrp(frp) {
  const value = Number(frp || 0);

  if (value >= 50) return "HIGH";
  if (value >= 10) return "MEDIUM";
  return "LOW";
}

function mapBackendEvent(event) {
  return {
    id: `FIRE-${String(event.event_id).padStart(3, "0")}`,
    backendId: event.event_id,
    lat: Number(event.latitude),
    lng: Number(event.longitude),
    location: "India",
    city: "",
    state: "",
    country: "India",
    type: formatEventType(event.predicted_class),
    risk: String(event.risk_level || "LOW").toUpperCase(),
    confidence: Math.round(Number(event.prediction_confidence || 0) * 100),
    satelliteConfidence: Math.round(Number(event.confidence_score || 0) * 100),
    persistence: formatPersistence(event.persistence_days),
    persistenceDays: Number(event.persistence_days || 0),
    intensity: intensityFromFrp(event.frp),
    frp: Number(event.frp || 0),
    facility:
      Number.isFinite(Number(event.distance_to_industry_km))
        ? "Industrial / mapped infrastructure"
        : "Mapped infrastructure",
    distance: formatDistance(event.distance_to_industry_km),
    distanceToIndustryKm: Number(event.distance_to_industry_km),
    distanceToPowerKm: Number(event.distance_to_power_km),
    distanceToOilGasKm: Number(event.distance_to_oil_gas_km),
    distanceToRoadKm: Number(event.distance_to_road_km),
    distanceToSettlementKm: Number(event.distance_to_settlement_km),
    detectionCount: Number(event.detection_count || 0),
    firstDetectedAt: event.first_detected_at,
    lastDetectedAt: event.last_detected_at,
    acqDatetime: event.acq_datetime,
    satellite: event.satellite,
    instrument: event.instrument,
    landcoverName: event.landcover_name,
    classificationStatus: event.classification_status,
    modelPredictedClass: event.model_predicted_class,
    predictionConfidence: Number(event.prediction_confidence || 0),
    riskScore: Number(event.risk_score || 0),
    decisionStatus: event.decision_status,
    riskReasons: event.risk_reasons || "No risk explanation available.",
    nearIndustry: Boolean(event.near_industry_5km),
    nearPower: Boolean(event.near_power_10km),
    nearOilGas: Boolean(event.near_oil_gas_10km),
    nearRoad: Boolean(event.near_road_1km),
    nearSettlement: Boolean(event.near_settlement_5km),
    industrialContext: Boolean(event.industrial_context),
    settlementContext: Boolean(event.settlement_context),
    roadContext: Boolean(event.road_context),
    vegetationContext: Boolean(event.vegetation_context),
    agricultureContext: Boolean(event.agriculture_context),
    builtupContext: Boolean(event.builtup_context),
    waterContext: Boolean(event.water_context),
    wetlandContext: Boolean(event.wetland_context),
    reason:
      event.risk_reasons ||
      "Risk assessment combines satellite thermal evidence, persistence, mapped infrastructure, land-cover context, and model classification.",
  };
}

function EventMap({ event }) {
  const mapRef = useRef(null);
  const containerRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current) return;

    if (mapRef.current) {
      mapRef.current.remove();
      mapRef.current = null;
    }

    const map = L.map(containerRef.current, {
      zoomControl: true,
      attributionControl: true,
      minZoom: 8,
      maxZoom: 18,
      zoomSnap: 0.5,
      zoomDelta: 0.5,
      wheelDebounceTime: 80,
      wheelPxPerZoomLevel: 180,
    }).setView([event.lat, event.lng], 12);

    mapRef.current = map;

    /*
      OpenStreetMap base layer
    */
    L.tileLayer(
      "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
      {
        maxZoom: 19,
        attribution: "&copy; OpenStreetMap contributors",
      }
    ).addTo(map);


    /*
      Thermal event marker
    */

    const eventIcon = L.divIcon({
      className: "thermal-map-marker",
      html: `
        <div class="thermal-marker-ring">
          <div class="thermal-marker-core"></div>
        </div>
      `,
      iconSize: [40, 40],
      iconAnchor: [20, 20],
    });


    L.marker(
      [event.lat, event.lng],
      {
        icon: eventIcon,
      }
    )
      .addTo(map)
      .bindPopup(`
        <div class="map-popup">
          <strong>${event.id}</strong>
          <span>${event.location}</span>
          <b>${event.risk} RISK</b>
        </div>
      `);


    /*
      Investigation radius
    */

    L.circle(
      [event.lat, event.lng],
      {
        radius: 1000,

        color: "#ff4b42",

        weight: 1,

        opacity: 0.8,

        fillColor: "#ff4035",

        fillOpacity: 0.08,
      }
    ).addTo(map);


    /*
      Nearby industrial facilities
      Demo coordinates for UI.
      Backend will replace these later.
    */

    const facilities = [
      {
        lat: event.lat + 0.009,
        lng: event.lng + 0.006,
        name: "Industrial Facility",
        distance: event.distance,
      },

      {
        lat: event.lat - 0.014,
        lng: event.lng + 0.011,
        name: "Processing Facility",
        distance: "2.1 km",
      },

      {
        lat: event.lat + 0.018,
        lng: event.lng - 0.012,
        name: "Power Infrastructure",
        distance: "3.4 km",
      },
    ];


    facilities.forEach((facility) => {

      const facilityIcon = L.divIcon({
        className: "facility-map-marker",

        html: `
          <div class="facility-marker">
            🏭
          </div>
        `,

        iconSize: [30, 30],

        iconAnchor: [15, 15],
      });


      L.marker(
        [facility.lat, facility.lng],
        {
          icon: facilityIcon,
        }
      )
        .addTo(map)
        .bindPopup(`
          <div class="map-popup">
            <strong>${facility.name}</strong>
            <span>Industrial infrastructure</span>
            <b>${facility.distance}</b>
          </div>
        `);

    });


    /*
      Event → nearest facility line
    */

    L.polyline(
      [
        [event.lat, event.lng],
        [facilities[0].lat, facilities[0].lng],
      ],
      {
        color: "#ff6b61",

        weight: 1,

        dashArray: "5, 7",

        opacity: 0.75,
      }
    ).addTo(map);


    /*
      Force correct map size
    */

    setTimeout(() => {
      map.invalidateSize();
    }, 200);


    return () => {
      map.remove();
      mapRef.current = null;
    };

  }, [event]);


  return (
    <div
      ref={containerRef}
      className="real-event-map"
    />
  );
}


function IndiaRiskMap({ events, selectedEvent, onSelectEvent }) {
  const mapRef = useRef(null);
  const containerRef = useRef(null);
  const markerLayerRef = useRef(null);

  // Create the map exactly once. The initial view is India-focused,
  // not event-focused. Event zooming happens only after a marker click.
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    const map = L.map(containerRef.current, {
      zoomControl: true,
      attributionControl: true,
      minZoom: 4,
      maxZoom: 14,
      maxBounds: L.latLngBounds(
        [5.5, 66.5],
        [38.5, 100]
      ),
      maxBoundsViscosity: 1,
      zoomSnap: 0.5,
      zoomDelta: 0.5,
      wheelDebounceTime: 80,
      wheelPxPerZoomLevel: 180,
    }).setView([22.5, 79.0], 5);

    mapRef.current = map;

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: "&copy; OpenStreetMap contributors",
    }).addTo(map);

    markerLayerRef.current = L.layerGroup().addTo(map);

    const resizeTimer = window.setTimeout(() => {
      map.invalidateSize({ pan: false });
    }, 150);

    return () => {
      window.clearTimeout(resizeTimer);
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
      markerLayerRef.current = null;
    };
  }, []);

  // Refresh markers without recreating the map or resetting its zoom.
  useEffect(() => {
    const map = mapRef.current;
    const layer = markerLayerRef.current;

    if (!map || !layer) return;

    layer.clearLayers();

    events
      .filter((event) => ["HIGH", "MEDIUM", "LOW"].includes(event.risk))
      .forEach((event) => {
        const riskClass = event.risk.toLowerCase();
        const isHigh = event.risk === "HIGH";
        const isSelected = selectedEvent?.id === event.id;

        const icon = L.divIcon({
          className: "india-risk-marker",
          html: `
            <div class="india-marker-shell ${riskClass} ${isSelected ? "selected" : ""}">
              <div class="india-marker-core"></div>
            </div>
          `,
          iconSize: [34, 34],
          iconAnchor: [17, 17],
        });

        const marker = L.marker([event.lat, event.lng], {
          icon,
          title: event.id,
        })
          .addTo(layer)
          .bindTooltip(
            `<div class="hotspot-tooltip">
              <strong>${event.location}</strong>
              <span>${event.id} · ${event.type}</span>
              <b class="${riskClass}">${event.risk} RISK</b>
              <small>${event.lat.toFixed(4)}° N, ${event.lng.toFixed(4)}° E</small>
            </div>`,
            {
              direction: "top",
              offset: [0, -16],
              opacity: 1,
              sticky: true,
              className: "hotspot-tooltip-wrap",
            }
          );

        marker.on("click", () => {
          // Only an explicit event click changes the map to a local view.
          map.flyTo([event.lat, event.lng], 12, {
            duration: 0.9,
            easeLinearity: 0.2,
          });
          onSelectEvent(event);
        });

        if (isHigh) {
          L.circle([event.lat, event.lng], {
            radius: isSelected ? 32000 : 18000,
            color: "#ff4438",
            weight: 1,
            opacity: isSelected ? 0.7 : 0.35,
            fillColor: "#ff4438",
            fillOpacity: isSelected ? 0.12 : 0.04,
            interactive: false,
          }).addTo(layer);
        }
      });

    const resizeTimer = window.setTimeout(() => {
      map.invalidateSize({ pan: false });
    }, 80);

    return () => window.clearTimeout(resizeTimer);
  }, [events, selectedEvent, onSelectEvent]);

  // Returning to Global View resets to the India overview.
  // Selecting an event does not trigger an automatic second zoom;
  // marker clicks are the only action that zooms into an event.
  useEffect(() => {
    if (!mapRef.current || selectedEvent) return;

    mapRef.current.setView([22.5, 79.0], 5, {
      animate: false,
    });
  }, [selectedEvent]);

  return (
    <div className="india-risk-map-wrap">
      <div className="india-map-header">
        <div>
          <span>INDIA THERMAL SURVEILLANCE</span>
          <strong>RISK EVENT MAP</strong>
        </div>

        <div className="india-map-legend">
          <span><i className="legend-high"></i> HIGH</span>
          <span><i className="legend-medium"></i> MEDIUM</span>
          <span><i className="legend-low"></i> LOW</span>
        </div>
      </div>

      <div ref={containerRef} className="india-risk-map"></div>

      <div className="india-map-status">
        <span className="map-live-dot"></span>
        ACTIVE EVENT OVERLAY
      </div>
    </div>
  );
}

function App() {
  const [selectedEvent, setSelectedEvent] = useState(null);
  const [panelOpen, setPanelOpen] = useState(false);
  const [investigationOpen, setInvestigationOpen] = useState(false);
  const [activeTab, setActiveTab] = useState("Overview");
  const [events, setEvents] = useState([]);
  const [eventsLoading, setEventsLoading] = useState(true);
  const [eventsError, setEventsError] = useState(null);
  const [ingestionStatus, setIngestionStatus] = useState(null);
  const [ingestionStatusUnavailable, setIngestionStatusUnavailable] = useState(false);
  const [dashboardView, setDashboardView] = useState("global");
  const [alertsOpen, setAlertsOpen] = useState(false);
  const [screenWidth, setScreenWidth] = useState(window.innerWidth);
  const [replayStep, setReplayStep] = useState(4);
  const [isReplaying, setIsReplaying] = useState(false);
  const geocodeCache = useRef(new Map());
  const alertMenuRef = useRef(null);


  const highRiskEvents = events.filter(
    (event) => event.risk === "HIGH"
  );
  const alertEvents = events
    .filter((event) => event.risk === "HIGH" || event.risk === "MEDIUM")
    .sort((first, second) => {
      if (first.risk === second.risk) return 0;
      return first.risk === "HIGH" ? -1 : 1;
    });
  const selectedAlertIndex = alertEvents.findIndex(
    (event) => event.id === selectedEvent?.id
  );
  const feedIndicator = getFeedIndicator(
    ingestionStatus,
    ingestionStatusUnavailable
  );

  useEffect(() => {
    if (!alertsOpen) return undefined;

    const closeOnOutsideClick = (event) => {
      if (!alertMenuRef.current?.contains(event.target)) {
        setAlertsOpen(false);
      }
    };
    const closeOnEscape = (event) => {
      if (event.key === "Escape") setAlertsOpen(false);
    };

    document.addEventListener("pointerdown", closeOnOutsideClick);
    document.addEventListener("keydown", closeOnEscape);

    return () => {
      document.removeEventListener("pointerdown", closeOnOutsideClick);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [alertsOpen]);

  useEffect(() => {
    let cancelled = false;

    const loadEvents = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/events`);

        if (!response.ok) {
          throw new Error(`Backend returned HTTP ${response.status}`);
        }

        const data = await response.json();

        if (!Array.isArray(data)) {
          throw new Error("Backend returned an invalid event payload.");
        }

        const mappedEvents = data.map(mapBackendEvent);

        if (!cancelled) {
          setEvents(mappedEvents);
          setEventsError(null);
        }
      } catch (error) {
        console.error("Failed to load GeoFlare events:", error);

        if (!cancelled) {
          setEventsError(error.message || "Unable to load events.");
        }
      } finally {
        if (!cancelled) {
          setEventsLoading(false);
        }
      }
    };

    loadEvents();
    const refreshTimer = setInterval(loadEvents, 60000);

    return () => {
      cancelled = true;
      clearInterval(refreshTimer);
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    let requestInFlight = false;

    const loadIngestionStatus = async () => {
      if (requestInFlight) return;
      requestInFlight = true;

      try {
        const response = await fetch(`${API_BASE_URL}/ingestion/status`);
        if (!response.ok) {
          throw new Error(`Backend returned HTTP ${response.status}`);
        }

        const status = await response.json();
        if (!cancelled) {
          setIngestionStatus(status);
          setIngestionStatusUnavailable(false);
        }
      } catch (error) {
        if (!cancelled) {
          setIngestionStatusUnavailable(true);
          console.warn("Unable to load ingestion status:", error);
        }
      } finally {
        requestInFlight = false;
      }
    };

    loadIngestionStatus();
    const refreshTimer = setInterval(loadIngestionStatus, 60000);

    return () => {
      cancelled = true;
      clearInterval(refreshTimer);
    };
  }, []);

  useEffect(() => {
    if (!selectedEvent || !Number.isFinite(selectedEvent.lat) || !Number.isFinite(selectedEvent.lng)) {
      return undefined;
    }

    const cacheKey = `${selectedEvent.lat.toFixed(4)},${selectedEvent.lng.toFixed(4)}`;
    const cachedAddress = geocodeCache.current.get(cacheKey);

    if (cachedAddress) {
      setSelectedEvent((currentEvent) =>
        currentEvent?.id === selectedEvent.id
          ? { ...currentEvent, ...cachedAddress }
          : currentEvent
      );
      return undefined;
    }

    const controller = new AbortController();

    fetch(
      `https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat=${selectedEvent.lat}&lon=${selectedEvent.lng}&zoom=10`,
      {
        signal: controller.signal,
        headers: { Accept: "application/json" },
      }
    )
      .then((response) => {
        if (!response.ok) throw new Error("Reverse geocoding failed");
        return response.json();
      })
      .then((result) => {
        const address = result.address || {};
        const locality = {
          city: address.city || address.town || address.village || address.municipality || "",
          state: address.state || address.region || "",
          country: address.country || selectedEvent.country || "India",
        };

        geocodeCache.current.set(cacheKey, locality);
        setSelectedEvent((currentEvent) =>
          currentEvent?.id === selectedEvent.id
            ? { ...currentEvent, ...locality }
            : currentEvent
        );
      })
      .catch((error) => {
        if (error.name !== "AbortError") {
          console.warn("Unable to resolve event locality:", error);
        }
      });

    return () => controller.abort();
  }, [selectedEvent?.id, selectedEvent?.lat, selectedEvent?.lng]);

  const openRiskEvent = (event) => {
    if (!event) return;
  setAlertsOpen(false);
  setSelectedEvent(event);
  setPanelOpen(true);
  setActiveTab("Overview");

};

const changeRiskEvent = (direction) => {
  if (!selectedEvent) return;

  const currentIndex = alertEvents.findIndex(
    (event) => event.id === selectedEvent.id
  );

  if (currentIndex === -1) {
    openRiskEvent(alertEvents[0]);
    return;
  }

  const nextIndex =
    (currentIndex + direction + alertEvents.length) %
    alertEvents.length;

  openRiskEvent(alertEvents[nextIndex]);
};

  const startReplay = () => {
    if (isReplaying) return;

    setIsReplaying(true);
    setReplayStep(0);

    let step = 0;

    const interval = setInterval(() => {
      step += 1;
      setReplayStep(step);

      if (step >= 4) {
        clearInterval(interval);
        setIsReplaying(false);
      }
    }, 900);
  };

  useEffect(() => {
    const handleResize = () => {
      setScreenWidth(window.innerWidth);
    };

    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
    };
  }, []);

  const openEvent = useCallback((event) => {
    if (!event) return;
    setAlertsOpen(false);
    setSelectedEvent(event);
    setPanelOpen(true);
  }, []);

  const openRiskEvents = () => {
    setAlertsOpen(false);
    setDashboardView("global");
    setPanelOpen(false);

    const firstHighRisk = events.find(
      (event) => event.risk === "HIGH"
    );
    const firstAlert = firstHighRisk || alertEvents[0];

    if (firstAlert) {
      setSelectedEvent(firstAlert);
      setPanelOpen(true);
      setActiveTab("Overview");
    } else if (events.length > 0) {
      setSelectedEvent(events[0]);
      setPanelOpen(true);
      setActiveTab("Overview");
    }
  };

  const openDashboardView = (view) => {
    setAlertsOpen(false);
    setPanelOpen(false);
    setSelectedEvent(null);
    setDashboardView(view);
  };

  const closePanel = () => {
    setAlertsOpen(false);
    setPanelOpen(false);
    setSelectedEvent(null);
  };

  const changeEvent = (direction) => {
    if (!selectedEvent) return;

    const currentIndex = events.findIndex(
      (event) => event.id === selectedEvent.id
    );

    let nextIndex =
      currentIndex + direction;

    if (nextIndex < 0) {
      nextIndex = events.length - 1;
    }

    if (nextIndex >= events.length) {
      nextIndex = 0;
    }

    const nextEvent = events[nextIndex];

    setSelectedEvent(nextEvent);

  };

  const focusIndia = () => {
    setAlertsOpen(false);
    setPanelOpen(false);
    setSelectedEvent(null);
    setDashboardView("global");
  };

  return (
    <div className="app">

      <header className="topbar">
        <div className="brand">
          <h1>GeoFlare</h1>
        </div>

        <nav className="navigation" aria-label="Primary navigation">
          <button
            className={`nav-button ${dashboardView === "global" ? "active" : ""}`}
            onClick={focusIndia}
          >
            <span className="nav-icon" aria-hidden="true">◉</span>
            Global View
          </button>
          <button className="nav-button risk-button" onClick={openRiskEvents}>
            <span className="nav-icon" aria-hidden="true">♨</span>
            Risk Events
          </button>
          <button
            className={`nav-button ${dashboardView === "facilities" ? "active" : ""}`}
            onClick={() => openDashboardView("facilities")}
          >
            <span className="nav-icon" aria-hidden="true">▦</span>
            Facilities
          </button>
          <button
            className={`nav-button ${dashboardView === "analytics" ? "active" : ""}`}
            onClick={() => openDashboardView("analytics")}
          >
            <span className="nav-icon" aria-hidden="true">▥</span>
            Analytics
          </button>
          <button
            className={`nav-button ${dashboardView === "reports" ? "active" : ""}`}
            onClick={() => openDashboardView("reports")}
          >
            <span className="nav-icon" aria-hidden="true">▤</span>
            Reports
          </button>
        </nav>

        <div className="alert-menu" ref={alertMenuRef}>
          <button
            className={`alert-menu-trigger ${alertsOpen ? "open" : ""}`}
            type="button"
            aria-expanded={alertsOpen}
            aria-controls="risk-alert-dropdown"
            aria-label={`Risk alerts, ${alertEvents.length} active`}
            disabled={panelOpen || investigationOpen}
            onClick={() => setAlertsOpen((isOpen) => !isOpen)}
          >
            <span className="alert-menu-icon" aria-hidden="true">!</span>
            <span>Alerts</span>
            <span className={`alert-menu-badge ${alertEvents.length ? "has-alerts" : ""}`}>
              {alertEvents.length}
            </span>
          </button>

          {alertsOpen && !panelOpen && !investigationOpen && (
            <div
              className="alert-dropdown"
              id="risk-alert-dropdown"
              aria-label="Active medium- and high-risk alerts"
              aria-live="polite"
            >
              <div className="alert-center-heading">
                <div className="alert-center-title">
                  <span className="alert-center-indicator" aria-hidden="true"></span>
                  <div>
                    <strong>Risk alerts</strong>
                    <small>MEDIUM &amp; HIGH</small>
                  </div>
                </div>
                <span className="alert-center-count">
                  {alertEvents.length} ACTIVE
                </span>
              </div>

              {eventsLoading ? (
                <p className="alert-dropdown-message">Loading alerts…</p>
              ) : eventsError ? (
                <p className="alert-dropdown-message">Alerts are unavailable while the event feed is offline.</p>
              ) : alertEvents.length === 0 ? (
                <p className="alert-dropdown-message">No medium or high risk events right now.</p>
              ) : (
                <div
                  className="alert-center-list"
                  aria-label="All active risk alerts"
                  tabIndex={0}
                >
                  {alertEvents.map((event) => (
                    <button
                      key={event.id}
                      className={`screen-alert ${event.risk.toLowerCase()}`}
                      onClick={() => openEvent(event)}
                      aria-label={`View ${event.risk.toLowerCase()} risk alert for ${event.id}`}
                    >
                      <span className="screen-alert-level">{event.risk} RISK</span>
                      <strong>{formatEventLocation(event)}</strong>
                      <small>
                        {event.type} · {event.confidence}% confidence
                      </small>
                      <span className="screen-alert-action">View event <span aria-hidden="true">→</span></span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        <div
          className={`live-status feed-status ${feedIndicator.tone}`}
          aria-label={`FIRMS data feed: ${feedIndicator.label}`}
          title={
            ingestionStatus?.last_success_at
              ? `Last successful ingestion: ${new Date(ingestionStatus.last_success_at).toLocaleString()}`
              : ingestionStatus?.last_attempt_at
                ? `Last ingestion attempt: ${new Date(ingestionStatus.last_attempt_at).toLocaleString()}`
                : "No ingestion attempt recorded since this backend started."
          }
        >
          <span className="status-dot"></span>
          <div className="feed-status-text">
            <strong>{feedIndicator.label}</strong>
            <small>
              {ingestionStatus?.last_success_at
                ? `Last success ${formatTimeAgo(ingestionStatus.last_success_at)}`
                : ingestionStatus?.last_attempt_at
                  ? `Last attempt ${formatTimeAgo(ingestionStatus.last_attempt_at)}`
                  : "No attempt yet"}
            </small>
          </div>
        </div>
      </header>

      {/* ================= GLOBE ================= */}

      <main
        className={
          panelOpen
            ? "globe-section panel-mode"
            : "globe-section"
        }
      >

        {eventsLoading && (
          <div className="instructions">
            Loading live thermal events...
          </div>
        )}

        {eventsError && !eventsLoading && (
          <div className="instructions">
            Backend connection error: {eventsError}
          </div>
        )}

        <IndiaRiskMap
          events={events}
          selectedEvent={selectedEvent}
          onSelectEvent={openEvent}
        />


      </main>


      {/* ================= EVENT PANEL ================= */}

      {panelOpen && selectedEvent && (





        <aside className="event-panel">
          <div className="risk-event-switcher">
  <button
    className="risk-switch-button"
    disabled={alertEvents.length === 0}
    onClick={() => changeRiskEvent(-1)}
  >
    ←
  </button>

  <div className="risk-event-counter">
    <span>RISK ALERTS</span>
    <strong>
      {selectedAlertIndex >= 0
        ? `${selectedAlertIndex + 1} / ${alertEvents.length}`
        : `${alertEvents.length} ACTIVE`}
    </strong>
  </div>

  <button
    className="risk-switch-button"
    disabled={alertEvents.length === 0}
    onClick={() => changeRiskEvent(1)}
  >
    →
  </button>
</div>

<div className="risk-summary-strip">
  <div>
    <span>HIGH RISK</span>
    <strong>{highRiskEvents.length}</strong>
  </div>

  <div>
    <span>MEDIUM RISK</span>
    <strong>{alertEvents.filter((event) => event.risk === "MEDIUM").length}</strong>
  </div>

  <div>
    <span>REGION</span>
    <strong>INDIA</strong>
  </div>
</div>
<div className="risk-list">
  <div className="risk-list-title">
    <span>ACTIVE RISK ALERTS</span>
    <small>{alertEvents.length} DETECTED</small>
  </div>

  {alertEvents.map((event) => (
    <button
      key={event.id}
      className={`risk-event-card ${
        selectedEvent?.id === event.id ? "selected" : ""
      }`}
      onClick={() => openRiskEvent(event)}
    >
      <div className="risk-event-dot"></div>

      <div className="risk-event-card-main">
        <div className="risk-event-card-top">
          <strong>{event.id}</strong>
          <span className={event.risk.toLowerCase()}>{event.risk}</span>
        </div>

        <div className="risk-event-location">
          {event.location}
        </div>

        <div className="risk-event-type">
          {event.type}
        </div>

        <div className="risk-event-meta">
          <span>{event.confidence}% CONF.</span>
          <span>{event.persistence}</span>
          <span>{event.distance}</span>
        </div>
      </div>

      <div className="risk-event-arrow">→</div>
    </button>
  ))}
</div>
          {/* PANEL HEADER */}

          <div className="panel-header">

            <button
              className="back-button"
              onClick={closePanel}
            >
              ← Back to Events
            </button>

            <button
              className="close-button"
              onClick={closePanel}
            >
              ×
            </button>

          </div>


          {/* EVENT SWITCHER */}

          <div className="event-switcher">

            <button
              onClick={() => changeEvent(-1)}
            >
              ←
            </button>

            <span>
              {events.findIndex(
                (event) =>
                  event.id === selectedEvent.id
              ) + 1}
              {" / "}
              {events.length}
            </span>

            <button
              onClick={() => changeEvent(1)}
            >
              →
            </button>

          </div>


          {/* RISK */}

          <div className="risk-badge">
            <span>●</span>
            {selectedEvent.risk} RISK
          </div>


          <div className="event-id">
            Event ID: {selectedEvent.id}
          </div>


          <h2>
            {selectedEvent.city || selectedEvent.state || selectedEvent.country || selectedEvent.location}
          </h2>

          <div className="event-locality">
            {[selectedEvent.city, selectedEvent.state, selectedEvent.country]
              .filter(Boolean)
              .join(", ") || "Resolving location..."}
          </div>


          <div className="coordinates">
            {selectedEvent.lat.toFixed(4)}° N
            {"  "}
            {selectedEvent.lng.toFixed(4)}° E
          </div>


          {/* TABS */}

          <div className="event-tabs">

  {[
    "Overview",
    "Imagery",
    "Timeline",
    "Context",
    "Facilities"
  ].map((tab) => (

    <button
      key={tab}
      className={activeTab === tab ? "active" : ""}
      onClick={() => setActiveTab(tab)}
    >
      {tab}
    </button>

  ))}

</div>


          {/* METRICS */}

          <div className="metrics-grid">

            <div className="metric-card">
              <span>◉</span>
              <small>CONFIDENCE</small>
              <strong>
                {selectedEvent.confidence}%
              </strong>
            </div>

            <div className="metric-card">
              <span>♨</span>
              <small>THERMAL INTENSITY</small>
              <strong>
                {selectedEvent.intensity}
              </strong>
            </div>

            <div className="metric-card">
              <span>◷</span>
              <small>PERSISTENCE</small>
              <strong>
                {selectedEvent.persistence}
              </strong>
            </div>

            <div className="metric-card">
              <span>●</span>
              <small>DISTANCE</small>
              <strong>
                {selectedEvent.distance}
              </strong>
            </div>

          </div>


          {/* EVENT TYPE */}

          <div className="event-type-card">

            <div className="fire-icon">
              ♨
            </div>

            <div>
              <small>EVENT TYPE</small>

              <strong>
                {selectedEvent.type}
              </strong>

              <p>
                Likely facility-related
                thermal source
              </p>
            </div>

          </div>


          {/* WHY */}

          <section className="why-section">

            <h3>
              WHY THIS EVENT MATTERS
            </h3>

            <p>
              {selectedEvent.reason}
            </p>

          </section>


          {/* ACTIONS */}

          <div className="action-buttons">

            <button
              className="investigate-button"
              onClick={() => {
                setAlertsOpen(false);
                setActiveTab("Imagery");
                setInvestigationOpen(true);
              }}
            >
              🔍 View Full Investigation →
            </button>

            <button className="track-button">
              ♧ Track This Event
            </button>

          </div>


          {/* RELATED EVENTS */}

          <section className="related-section">

            <div className="section-title">
              <span>RELATED EVENTS NEARBY</span>
              <button>See All →</button>
            </div>

            <div className="related-events">

              {events
                .filter(
                  (event) =>
                    event.id !== selectedEvent.id
                )
                .slice(0, 3)
                .map((event) => (

                  <button
                    key={event.id}
                    className="related-card"
                    onClick={() => openEvent(event)}
                  >

                    <div className="mini-map">
                      ●
                    </div>

                    <div>
                      <strong>
                        {event.id}
                      </strong>

                      <span>
                        {event.location}
                      </span>

                      <em
                        className={
                          event.risk === "HIGH"
                            ? "high-text"
                            : "medium-text"
                        }
                      >
                        {event.risk}
                      </em>
                    </div>

                    <span className="arrow">
                      →
                    </span>

                  </button>

                ))}

            </div>

          </section>


          <div className="source">
            Source: NASA FIRMS
            <span>|</span>
            Satellite Thermal Data
          </div>

        </aside>





      )}
      {investigationOpen && selectedEvent && (
  <div className="investigation-screen">

    {/* HEADER */}

    <div className="investigation-header">

      <button
        className="investigation-back"
        onClick={() => setInvestigationOpen(false)}
      >
        ← Back to Events
      </button>

      <div className="investigation-title">
        <span>EVENT INVESTIGATION</span>
        <strong>{selectedEvent.id}</strong>
      </div>

      <div className="investigation-risk">
        ● {selectedEvent.risk} RISK
      </div>

    </div>


    {/* MAIN CONTENT */}

    <div className="investigation-content">

      {/* LEFT */}

      <section className="investigation-left">
        {activeTab === "Imagery" && (
  <div className="tab-view">

    <div className="tab-view-header">
      <div>
        <span>SATELLITE IMAGERY</span>
        <h2>Thermal Observation</h2>
      </div>

      <small>NASA FIRMS</small>
    </div>

    <div className="real-map-wrapper">

  <div className="map-top-bar">

    <div>
      <span>SATELLITE / GEOGRAPHIC VIEW</span>
    </div>

    <div className="map-source">
      OSM
    </div>

  </div>

  <EventMap event={selectedEvent} />

  <div className="map-overlay-info">

    <div className="map-status">
      <span className="map-live-dot"></span>
      EVENT DETECTION
    </div>

    <div className="map-coordinates-live">
      {selectedEvent.lat.toFixed(4)}° N&nbsp;&nbsp;
      {selectedEvent.lng.toFixed(4)}° E
    </div>

  </div>

</div>

    <div className="imagery-info">

      <div>
        <span>OBSERVATION SOURCE</span>
        <strong>NASA FIRMS</strong>
      </div>

      <div>
        <span>THERMAL INTENSITY</span>
        <strong>{selectedEvent.intensity}</strong>
      </div>

      <div>
        <span>CONFIDENCE</span>
        <strong>{selectedEvent.confidence}%</strong>
      </div>

    </div>

  </div>
)}
{activeTab === "Timeline" && (
  <div className="tab-view">

    <div className="tab-view-header">
      <div>
        <span>OBSERVATION HISTORY</span>
        <h2>Event Timeline</h2>
      </div>

      <small>{selectedEvent.persistence}</small>
    </div>

    <div className="full-timeline">

      <div className="timeline-event">
        <div className="timeline-dot"></div>

        <div>
          <strong>
            {new Date(selectedEvent.firstDetectedAt).toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
            })}
          </strong>
          <h3>First Detection</h3>
          <p>
            Initial thermal anomaly detected by {selectedEvent.satellite || "NASA FIRMS"}.
          </p>
        </div>
      </div>

      <div className="timeline-event">
        <div className="timeline-dot"></div>

        <div>
          <strong>{selectedEvent.detectionCount}</strong>
          <h3>Satellite Observations</h3>
          <p>
            {selectedEvent.detectionCount} observation
            {selectedEvent.detectionCount === 1 ? "" : "s"} associated with this event.
          </p>
        </div>
      </div>

      <div className="timeline-event">
        <div className="timeline-dot"></div>

        <div>
          <strong>{selectedEvent.persistence}</strong>
          <h3>Estimated Persistence</h3>
          <p>
            Repeated thermal detections indicate an estimated activity episode over this period.
          </p>
        </div>
      </div>

      <div className="timeline-event">
        <div className="timeline-dot latest"></div>

        <div>
          <strong>
            {new Date(selectedEvent.lastDetectedAt).toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
            })}
          </strong>
          <h3>Latest Observation</h3>
          <p>
            Most recent satellite observation currently associated with this event.
          </p>
        </div>
      </div>

    </div>

  </div>
)}
{activeTab === "Context" && (
  <div className="tab-view">

    <div className="tab-view-header">
      <div>
        <span>GEOGRAPHIC ANALYSIS</span>
        <h2>Environmental Context</h2>
      </div>

      <small>OSM + LAND COVER</small>
    </div>

    <div className="context-map">

      <div className="context-grid"></div>

      <div className="context-zone industrial-zone">
        INDUSTRIAL
      </div>

      <div className="context-zone urban-zone">
        URBAN
      </div>

      <div className="context-zone agricultural-zone">
        AGRICULTURAL
      </div>

      <div className="context-hotspot">
        ●
      </div>

    </div>

    <div className="land-use">

      <div className="land-row">
        <span>Industrial / Power</span>
        <div className="land-bar">
          <i style={{ width: `${selectedEvent.industrialContext ? 100 : 20}%` }}></i>
        </div>
        <strong>{selectedEvent.industrialContext ? "YES" : "NO"}</strong>
      </div>

      <div className="land-row">
        <span>Agricultural</span>
        <div className="land-bar">
          <i style={{ width: `${selectedEvent.agricultureContext ? 100 : 20}%` }}></i>
        </div>
        <strong>{selectedEvent.agricultureContext ? "YES" : "NO"}</strong>
      </div>

      <div className="land-row">
        <span>Built-up</span>
        <div className="land-bar">
          <i style={{ width: `${selectedEvent.builtupContext ? 100 : 20}%` }}></i>
        </div>
        <strong>{selectedEvent.builtupContext ? "YES" : "NO"}</strong>
      </div>

      <div className="land-row">
        <span>Land cover</span>
        <div className="land-bar">
          <i style={{ width: "100%" }}></i>
        </div>
        <strong>{selectedEvent.landcoverName}</strong>
      </div>

    </div>

  </div>
)}
{activeTab === "Facilities" && (
  <div className="tab-view">

    <div className="tab-view-header">
      <div>
        <span>OPENSTREETMAP ANALYSIS</span>
        <h2>Nearby Facilities</h2>
      </div>

      <small>OSM</small>
    </div>

    <div className="facility-list">

      <div className="facility-item">

        <div className="facility-icon">
          🏭
        </div>

        <div>
          <strong>
            Industrial Facility
          </strong>

          <span>
            Manufacturing / Processing
          </span>
        </div>

        <b>
          {selectedEvent.distance}
        </b>

      </div>


      <div className="facility-item">

        <div className="facility-icon">
          🏭
        </div>

        <div>
          <strong>
            Processing Facility
          </strong>

          <span>
            Industrial infrastructure
          </span>
        </div>

        <b>
          2.1 km
        </b>

      </div>


      <div className="facility-item">

        <div className="facility-icon">
          ⚡
        </div>

        <div>
          <strong>
            Power Infrastructure
          </strong>

          <span>
            Energy facility
          </span>
        </div>

        <b>
          3.4 km
        </b>

      </div>

    </div>


    <div className="facility-note">

      <span>ⓘ</span>

      <p>
        Facility proximity is used as contextual
        evidence in the risk assessment. It does
        not independently confirm the cause of
        the thermal anomaly.
      </p>

    </div>

  </div>
)}


        {activeTab === "Overview" && (
        <>
        <div className="location-heading">

          <div>
            <span>THERMAL EVENT</span>

            <h1>
              {selectedEvent.location}
            </h1>

            <p>
              {selectedEvent.lat.toFixed(4)}° N
              {"  "}
              {selectedEvent.lng.toFixed(4)}° E
            </p>
          </div>

        </div>


        {/* SATELLITE IMAGE */}

        <div className="satellite-card">

          <div className="card-heading">
            <span>SATELLITE THERMAL OBSERVATION</span>

            <small>
              NASA FIRMS
            </small>
          </div>

          <div className="satellite-placeholder">

            <div className="satellite-grid"></div>

            <div className="thermal-center">
              <div className="thermal-ring"></div>
              <div className="thermal-dot"></div>
            </div>

            <div className="map-label label-1">
              INDUSTRIAL AREA
            </div>

            <div className="map-label label-2">
              DETECTION
            </div>

            <div className="map-coordinates">
              {selectedEvent.lat.toFixed(4)}° N
              {"  "}
              {selectedEvent.lng.toFixed(4)}° E
            </div>

          </div>

        </div>


        {/* TIMELINE */}

        <div className="timeline-card">


          <div className="card-heading">
            <span>EVENT TIMELINE</span>

            <small>
              {selectedEvent.persistence}
            </small>
          </div>
          <div className="timeline-replay">

            <div className="replay-header">
              <div>
                <span>THERMAL OBSERVATION SEQUENCE</span>
                <strong>EVENT REPLAY</strong>
              </div>

              <div className="replay-status">
                <span className="replay-dot"></span>
                {selectedEvent.persistence} OBSERVED
              </div>
            </div>

            <div className="replay-track">

              <div className="replay-line"></div>

              <div className={`replay-step ${replayStep >= 1 ? "active" : ""}`}>
                <div className="replay-point"></div>
                <div className="replay-info">
                  <strong>
                    {new Date(selectedEvent.firstDetectedAt).toLocaleTimeString([], {
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </strong>
                  <span>DETECTION</span>
                  <small>Initial thermal anomaly</small>
                </div>
              </div>

              <div className={`replay-step ${replayStep >= 2 ? "active" : ""}`}>
                <div className="replay-point"></div>
                <div className="replay-info">
                  <strong>{selectedEvent.detectionCount} OBS.</strong>
                  <span>PERSISTENCE</span>
                  <small>Thermal source remains associated with this event</small>
                </div>
              </div>

              <div className={`replay-step ${replayStep >= 3 ? "active" : ""}`}>
                <div className="replay-point"></div>
                <div className="replay-info">
                  <strong>{selectedEvent.intensity}</strong>
                  <span>THERMAL INTENSITY</span>
                  <small>Latest FIRMS FRP: {selectedEvent.frp.toFixed(2)} MW</small>
                </div>
              </div>

              <div className={`replay-step ${replayStep >= 4 ? "active latest" : ""}`}>
                <div className="replay-point"></div>
                <div className="replay-info">
                  <strong>
                    {new Date(selectedEvent.lastDetectedAt).toLocaleTimeString([], {
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </strong>
                  <span>LATEST</span>
                  <small>Most recent observation</small>
                </div>
              </div>

            </div>

            <div className="replay-controls">

              <button
                className="replay-button"
                onClick={startReplay}
                disabled={isReplaying}
              >
                {isReplaying ? "■" : "▶"}
              </button>

              <div className="replay-progress">
                <div
                  className="replay-progress-fill"
                  style={{
                    width: `${(replayStep / 4) * 100}%`,
                  }}
                ></div>
              </div>

              <span className="replay-time">
                {replayStep <= 1 &&
                  new Date(selectedEvent.firstDetectedAt).toLocaleTimeString([], {
                    hour: "2-digit",
                    minute: "2-digit",
                  })}
                {replayStep === 2 && `${selectedEvent.detectionCount} OBS.`}
                {replayStep === 3 && selectedEvent.intensity}
                {replayStep === 4 &&
                  new Date(selectedEvent.lastDetectedAt).toLocaleTimeString([], {
                    hour: "2-digit",
                    minute: "2-digit",
                  })}
              </span>

            </div>

          </div>

          </div>

        <div className="risk-explanation-card">
  <div className="risk-explanation-header">
    <div>
      <span>MODEL ASSESSMENT</span>
      <h3>WHY HIGH RISK?</h3>
    </div>

    <div className="risk-score">
      <strong>{selectedEvent.risk}</strong>
      <small>RISK LEVEL</small>
    </div>
  </div>

  <div className="risk-factors">

    <div className="risk-factor">
      <div className="factor-icon">◉</div>

      <div className="factor-content">
        <div className="factor-title">
          <span>THERMAL INTENSITY</span>
          <strong>{selectedEvent.intensity}</strong>
        </div>

        <div className="factor-bar">
          <div
            className="factor-fill"
            style={{
              width:
                selectedEvent.intensity === "HIGH"
                  ? "92%"
                  : "58%",
            }}
          ></div>
        </div>

        <p>
          Strong thermal anomaly detected in satellite observation.
        </p>
      </div>
    </div>


    <div className="risk-factor">
      <div className="factor-icon">◷</div>

      <div className="factor-content">
        <div className="factor-title">
          <span>PERSISTENCE</span>
          <strong>{selectedEvent.persistence}</strong>
        </div>

        <div className="factor-bar">
          <div
            className="factor-fill"
            style={{
              width:
                parseFloat(selectedEvent.persistence) >= 4
                  ? "88%"
                  : "55%",
            }}
          ></div>
        </div>

        <p>
          Repeated thermal observations increase event significance.
        </p>
      </div>
    </div>


    <div className="risk-factor">
      <div className="factor-icon">⌖</div>

      <div className="factor-content">
        <div className="factor-title">
          <span>FACILITY PROXIMITY</span>
          <strong>{selectedEvent.distance}</strong>
        </div>

        <div className="factor-bar">
          <div
            className="factor-fill"
            style={{
              width:
                parseFloat(selectedEvent.distance) <= 1.5
                  ? "94%"
                  : "60%",
            }}
          ></div>
        </div>

        <p>
          Thermal source is located near mapped infrastructure.
        </p>
      </div>
    </div>


    <div className="risk-factor">
      <div className="factor-icon">◎</div>

      <div className="factor-content">
        <div className="factor-title">
          <span>CLASSIFICATION CONFIDENCE</span>
          <strong>{selectedEvent.confidence}%</strong>
        </div>

        <div className="factor-bar">
          <div
            className="factor-fill"
            style={{
              width: `${selectedEvent.confidence}%`,
            }}
          ></div>
        </div>

        <p>
          Model confidence based on available geographic and temporal evidence.
        </p>
      </div>
    </div>

  </div>

  <div className="risk-conclusion">
    <span>ASSESSMENT</span>
    <p>
      Multiple independent indicators support a
      <strong> probable high-risk thermal event</strong>.
    </p>
  </div>
</div>

        </>
        )}


      </section>


      {/* RIGHT */}

      <section className="investigation-right">

        {/* SUMMARY */}

        <div className="investigation-card">

          <div className="card-heading">
            <span>EVENT SUMMARY</span>
          </div>

          <div className="summary-grid">

            <div>
              <small>CONFIDENCE</small>
              <strong>
                {selectedEvent.confidence}%
              </strong>
            </div>

            <div>
              <small>PERSISTENCE</small>
              <strong>
                {selectedEvent.persistence}
              </strong>
            </div>

            <div>
              <small>THERMAL INTENSITY</small>
              <strong>
                {selectedEvent.intensity}
              </strong>
            </div>

            <div>
              <small>FACILITY DISTANCE</small>
              <strong>
                {selectedEvent.distance}
              </strong>
            </div>

          </div>

        </div>


        {/* CLASSIFICATION */}

        <div className="investigation-card">

          <div className="card-heading">
            <span>AI CLASSIFICATION</span>
          </div>

          <div className="classification">

            <div className="classification-icon">
              ♨
            </div>

            <div>

              <strong>
                {selectedEvent.type}
              </strong>

              <p>
                {selectedEvent.classificationStatus === "CLASSIFIED"
                  ? `Model classification confidence: ${selectedEvent.confidence}%`
                  : "Classification requires review"}
              </p>

            </div>

            <div className="classification-score">
              {selectedEvent.confidence}%
            </div>

          </div>

        </div>


        {/* WHY HIGH RISK */}

        <div className="investigation-card why-high-risk">

          <div className="card-heading">
            <span>WHY HIGH RISK?</span>
          </div>

          <div className="evidence-list">

            <div className="evidence-item">
              <span>✓</span>
              <div>
                <strong>Persistent signal</strong>
                <small>
                  Detected repeatedly over
                  {selectedEvent.persistence}
                </small>
              </div>
            </div>

            <div className="evidence-item">
              <span>✓</span>
              <div>
                <strong>Industrial proximity</strong>
                <small>
                  Nearest facility:
                  {selectedEvent.distance}
                </small>
              </div>
            </div>

            <div className="evidence-item">
              <span>✓</span>
              <div>
                <strong>High thermal intensity</strong>
                <small>
                  Satellite observation classified
                  as high intensity
                </small>
              </div>
            </div>

            <div className="evidence-item">
              <span>✓</span>
              <div>
                <strong>Multiple observations</strong>
                <small>
                  Consistent thermal detections
                  strengthen the assessment
                </small>
              </div>
            </div>

          </div>

        </div>


        {/* CONTEXT */}

        <div className="investigation-card">

          <div className="card-heading">
            <span>GEOGRAPHIC CONTEXT</span>
          </div>

          <div className="context-list">

            <div>
              <span>Land context</span>
              <strong>{selectedEvent.landcoverName}</strong>
            </div>

            <div>
              <span>Nearest facility</span>
              <strong>
                {selectedEvent.facility}
              </strong>
            </div>

            <div>
              <span>Distance</span>
              <strong>
                {selectedEvent.distance}
              </strong>
            </div>

            <div>
              <span>Source</span>
              <strong>NASA FIRMS</strong>
            </div>

          </div>

        </div>


        {/* ACTIONS */}

        <div className="investigation-actions">

          <button className="track-large">
            ♧ Track This Event
          </button>

          <button
            className="return-large"
            onClick={() => setInvestigationOpen(false)}
          >
            ← Return to Events
          </button>

        </div>

      </section>

    </div>


    {/* FOOTER */}

    <div className="investigation-footer">

      <span>
        GeoFlare
      </span>

      <span>
        Decision-support assessment •
        Satellite-derived thermal observation
      </span>

      <span>
        Source: NASA FIRMS
      </span>

    </div>

  </div>
)}


      {dashboardView !== "global" && !panelOpen && (
        <aside className="dashboard-panel">
          <div className="dashboard-panel-header">
            <div>
              <span>GEOSPATIAL THERMAL INTELLIGENCE</span>
              <h2>
                {dashboardView === "facilities" && "Facilities"}
                {dashboardView === "analytics" && "Analytics"}
                {dashboardView === "reports" && "Reports"}
              </h2>
            </div>

            <button
              className="dashboard-panel-close"
              onClick={() => setDashboardView("global")}
              aria-label="Close section"
            >
              ×
            </button>
          </div>

          {dashboardView === "facilities" && (
            <div className="dashboard-panel-content">
              <div className="dashboard-panel-summary">
                <strong>OSM CONTEXT</strong>
                <span>Infrastructure proximity associated with active thermal events.</span>
              </div>

              {events
                .slice()
                .sort((a, b) => a.distanceToIndustryKm - b.distanceToIndustryKm)
                .slice(0, 8)
                .map((event) => (
                  <button
                    key={event.id}
                    className="dashboard-data-row"
                    onClick={() => openEvent(event)}
                  >
                    <div>
                      <strong>{event.id}</strong>
                      <span>{event.type}</span>
                    </div>
                    <div>
                      <b>{formatDistance(event.distanceToIndustryKm)}</b>
                      <small>INDUSTRY</small>
                    </div>
                  </button>
                ))}
            </div>
          )}

          {dashboardView === "analytics" && (
            <div className="dashboard-panel-content">
              <div className="analytics-grid">
                <div>
                  <span>ACTIVE EVENTS</span>
                  <strong>{events.length}</strong>
                </div>
                <div>
                  <span>HIGH RISK</span>
                  <strong>{highRiskEvents.length}</strong>
                </div>
                <div>
                  <span>PERSISTENT</span>
                  <strong>{events.filter((event) => event.persistenceDays >= 1).length}</strong>
                </div>
                <div>
                  <span>CLASSIFIED</span>
                  <strong>{events.filter((event) => event.classificationStatus === "CLASSIFIED").length}</strong>
                </div>
              </div>

              <div className="dashboard-panel-summary">
                <strong>CLASSIFICATION DISTRIBUTION</strong>
                <span>
                  Current active-event classifications from the GeoFlare ML pipeline.
                </span>
              </div>

              {[
                ["Industrial Fire", "industrial_fire"],
                ["Natural Fire", "natural_fire"],
                ["Agricultural Burning", "agricultural_burning"],
                ["Persistent Thermal Source", "persistent_thermal_source"],
                ["Uncertain", "uncertain"],
              ].map(([label, value]) => {
                const count = events.filter(
                  (event) => event.modelPredictedClass === value
                ).length;
                const percentage = events.length
                  ? Math.round((count / events.length) * 100)
                  : 0;

                return (
                  <div className="analytics-row" key={value}>
                    <div>
                      <span>{label}</span>
                      <strong>{count}</strong>
                    </div>
                    <div className="analytics-bar">
                      <i style={{ width: `${percentage}%` }}></i>
                    </div>
                    <small>{percentage}%</small>
                  </div>
                );
              })}
            </div>
          )}

          {dashboardView === "reports" && (
            <div className="dashboard-panel-content">
              <div className="report-card">
                <span>LIVE DATASET</span>
                <strong>NASA FIRMS + OSM + WorldCover</strong>
                <p>
                  Active thermal event intelligence currently available through the GeoFlare API.
                </p>
              </div>

              <div className="report-card">
                <span>EVENT COVERAGE</span>
                <strong>{events.length} active event episodes</strong>
                <p>
                  Each event is associated with satellite observations and spatial context.
                </p>
              </div>

              <div className="report-card">
                <span>MODEL STATUS</span>
                <strong>
                  {events.filter((event) => event.classificationStatus === "CLASSIFIED").length} classified
                </strong>
                <p>
                  Classification status is based on the configured model confidence threshold.
                </p>
              </div>

              <div className="report-note">
                Decision-support assessment. Satellite-derived observations do not independently confirm fire cause.
              </div>
            </div>
          )}
        </aside>
      )}

      {/* ================= BOTTOM STATS ================= */}

      {!panelOpen && dashboardView === "global" && (

        <div className="bottom-stats">

          <div className="stat-card">
            <span>ACTIVE EVENTS</span>
            <strong>{events.length}</strong>
            <small>LIVE EVENT EPISODES</small>
          </div>

          <div className="stat-card">
            <span>HIGH RISK</span>
            <strong>{String(highRiskEvents.length).padStart(2, "0")}</strong>
            <small>CURRENT HIGH-RISK</small>
          </div>

          <div className="stat-card">
            <span>PERSISTENT</span>
            <strong>{events.filter((event) => event.persistenceDays >= 1).length}</strong>
            <small>ESTIMATED PERSISTENCE</small>
          </div>

          <div className="stat-card">
            <span>INDUSTRIAL</span>
            <strong>{events.filter((event) => event.modelPredictedClass === "industrial_fire").length}</strong>
            <small>MODEL CLASSIFIED</small>
          </div>

        </div>

      )}


      {/* INSTRUCTIONS */}

      {!panelOpen && dashboardView === "global" && !eventsLoading && !eventsError && (

        <div className="instructions">
          Pan to explore • Scroll to zoom • Click an event to focus
        </div>

      )}

    </div>
  );
}

export default App;
