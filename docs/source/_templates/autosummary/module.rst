{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}

   {% block functions %}
   {% if functions %}
   .. rubric:: Functions

   .. autosummary::
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block classes %}
   {% if classes %}
   .. rubric:: Classes

   .. autosummary::
   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}
   .. rubric:: Exceptions

   .. autosummary::
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}


{% if functions %}
|

Function Descriptions
---------------------

{% for item in functions %}
      .. autofunction:: {{ item }}

{% raw %}
{% endraw %}
{%- endfor %}
{% endif %}


{% if classes %}
|

Class Descriptions
------------------

{% for item in classes %}
      .. autoclass:: {{ item }}
	 :members:

{% raw %}
{% endraw %}
{%- endfor %}
{% endif %}


{% if exceptions %}
|

Exception Descriptions
----------------------

{% for item in exceptions %}
      .. autoexception:: {{ item }}

{% raw %}
{% endraw %}
{%- endfor %}
{% endif %}
